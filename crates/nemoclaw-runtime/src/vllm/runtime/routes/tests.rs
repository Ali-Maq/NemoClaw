// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::hardware::{GIB, ProtectionPolicy};
use http_body_util::BodyExt;
use process_wrap::tokio::{CommandWrap, KillOnDrop, ProcessGroup};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const KEY: &str = "Bearer fixture-key";
const NOT_FOUND: &str = r#"{"error":"Not found"}"#;
/// The requests NemoClaw routes to vLLM, listed independently of the filter.
const INFERENCE: [(&str, &str); 7] = [
    ("GET", "/health"),
    ("GET", "/v1/models"),
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/completions"),
    ("POST", "/v1/responses"),
    ("POST", "/v1/messages"),
    ("POST", "/v1/messages/count_tokens"),
];

/// A request as the stand-in vLLM received it.
struct Seen {
    method: String,
    target: String,
    headers: hyper::HeaderMap,
    body: Vec<u8>,
}

/// A stand-in for vLLM that records every connection and request reaching it.
/// Its replies carry hop-by-hop headers that must not reach the client.
struct Upstream {
    address: SocketAddr,
    connections: Arc<AtomicUsize>,
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl Upstream {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let connections = Arc::new(AtomicUsize::new(0));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (accepted, recorded) = (connections.clone(), seen.clone());
        tokio::spawn(async move {
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                accepted.fetch_add(1, Ordering::SeqCst);
                let recorded = recorded.clone();
                let service = service_fn(move |request: Request<Incoming>| {
                    let recorded = recorded.clone();
                    async move {
                        let (parts, body) = request.into_parts();
                        // A request whose body the filter could not forward is
                        // incomplete and not recorded.
                        let Ok(body) = body.collect().await else {
                            return Ok(Response::new(Full::new(Bytes::new())));
                        };
                        let body = body.to_bytes().to_vec();
                        recorded.lock().unwrap().push(Seen {
                            method: parts.method.to_string(),
                            target: parts.uri.to_string(),
                            headers: parts.headers,
                            body,
                        });
                        let response = Response::builder()
                            .header("content-type", "application/json")
                            .header("connection", "x-upstream-hop")
                            .header("x-upstream-hop", "1")
                            .header("keep-alive", "timeout=5")
                            .header("proxy-authenticate", "Basic")
                            .header("trailer", "x-checksum")
                            .header("upgrade", "fixture")
                            .header("x-upstream", "kept")
                            .body(Full::new(Bytes::from_static(br#"{"ok":true}"#)))
                            .unwrap();
                        Ok::<_, Infallible>(response)
                    }
                });
                tokio::spawn(http1::Builder::new().serve_connection(TokioIo::new(stream), service));
            }
        });
        Self {
            address,
            connections,
            seen,
        }
    }

    /// Each request vLLM received, as `METHOD target`.
    fn requests(&self) -> Vec<String> {
        let seen = self.seen.lock().unwrap();
        seen.iter()
            .map(|request| format!("{} {}", request.method, request.target))
            .collect()
    }
}

/// Serve the filter in front of `backend` on an ephemeral loopback port.
async fn filter(backend: SocketAddr, header_timeout: Duration) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(serve(listener, backend, header_timeout, CONNECTIONS));
    address
}

/// Open a connection, have one request served on it, and keep it open, idle.
async fn keep_alive(address: SocketAddr) -> TcpStream {
    let mut stream = TcpStream::connect(address).await.unwrap();
    stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        .await
        .unwrap();
    read_response(&mut stream, r#"{"ok":true}"#).await;
    stream
}

/// Write raw bytes on a new connection and read until the filter closes it.
async fn exchange(address: SocketAddr, request: &str) -> String {
    let mut stream = TcpStream::connect(address).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    let mut response = Vec::new();
    // A reset after rejecting malformed input still ends the exchange.
    let _ = tokio::time::timeout(Duration::from_secs(5), stream.read_to_end(&mut response))
        .await
        .expect("the filter closes the connection");
    String::from_utf8(response).unwrap()
}

/// Read from a connection the filter keeps open until a response ends with `body`.
async fn read_response(stream: &mut TcpStream, body: &str) -> String {
    let mut response = Vec::new();
    tokio::time::timeout(Duration::from_secs(5), async {
        while !response.ends_with(body.as_bytes()) {
            let mut chunk = [0; 1024];
            let read = stream.read(&mut chunk).await.unwrap();
            assert_ne!(read, 0, "{}", String::from_utf8_lossy(&response));
            response.extend_from_slice(&chunk[..read]);
        }
    })
    .await
    .expect("the filter responds");
    String::from_utf8(response).unwrap()
}

fn request(method: &str, target: &str) -> String {
    format!(
        "{method} {target} HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: {KEY}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    )
}

fn statuses(response: &str) -> Vec<&str> {
    response
        .match_indices("HTTP/1.1 ")
        .map(|(index, _)| &response[index + 9..index + 12])
        .collect()
}

/// The header names of the first response, lowercase.
fn header_names(response: &str) -> Vec<String> {
    let head = response.split("\r\n\r\n").next().unwrap();
    head.lines()
        .skip(1)
        .filter_map(|line| line.split_once(':'))
        .map(|(name, _)| name.to_ascii_lowercase())
        .collect()
}

fn assert_not_found(response: &str, probe: &str) {
    assert_eq!(statuses(response), ["404"], "{probe}: {response}");
    assert!(
        response.contains("content-type: application/json\r\n"),
        "{probe}: {response}"
    );
    assert!(response.ends_with(NOT_FOUND), "{probe}: {response}");
}

#[test]
fn only_exact_origin_form_inference_requests_match() {
    let matches = |method: &str, target: &str, upgrade: bool| {
        let mut headers = HeaderMap::new();
        if upgrade {
            headers.insert(header::UPGRADE, HeaderValue::from_static("websocket"));
        }
        let method = Method::from_bytes(method.as_bytes()).unwrap();
        allowed(&method, &target.parse().unwrap(), &headers)
    };
    for (method, path) in INFERENCE {
        assert!(matches(method, path, false), "{method} {path}");
        assert!(matches(method, &format!("{path}?stream=true"), false));
        let other = if method == "GET" { "POST" } else { "GET" };
        for (method, target, upgrade) in [
            (method, path.to_owned(), true),
            (other, path.to_owned(), false),
            ("HEAD", path.to_owned(), false),
            ("OPTIONS", path.to_owned(), false),
            (&method.to_lowercase(), path.to_owned(), false),
            (method, format!("{path}/"), false),
            (method, format!("/{path}"), false),
            (method, path.to_uppercase(), false),
            (method, path.replacen('/', "/./", 1), false),
            (method, path.replace('e', "%65"), false),
            (method, format!("http://127.0.0.1:18888{path}"), false),
        ] {
            assert!(
                !matches(method, &target, upgrade),
                "{method} {target} upgrade: {upgrade}"
            );
        }
    }
    for (method, target) in [
        ("POST", "/invocations"),
        ("GET", "/v1/models/fixture"),
        ("POST", "/v1/load_lora_adapter"),
        ("POST", "/v1/chat/completions/../../invocations"),
        ("CONNECT", "127.0.0.1:18888"),
        ("OPTIONS", "*"),
        ("GET", "*"),
    ] {
        assert!(!matches(method, target, false), "{method} {target}");
    }
}

#[tokio::test]
async fn inference_requests_reach_vllm_with_their_key_query_and_body() {
    let upstream = Upstream::start().await;
    let address = filter(upstream.address, HEADER_TIMEOUT).await;
    for (method, path) in INFERENCE {
        let target = format!("{path}?stream=true&next=%2Finvocations");
        let body = if method == "POST" {
            r#"{"model":"m"}"#
        } else {
            ""
        };
        let response = exchange(
            address,
            &format!(
                "{method} {target} HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: {KEY}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            ),
        )
        .await;
        assert_eq!(statuses(&response), ["200"], "{method} {path}: {response}");
        assert!(response.ends_with(r#"{"ok":true}"#), "{response}");
        let seen = upstream.seen.lock().unwrap();
        let request = seen.last().unwrap();
        assert_eq!(
            (request.method.as_str(), request.target.as_str()),
            (method, target.as_str())
        );
        assert_eq!(request.headers["authorization"], KEY);
        assert_eq!(request.body, body.as_bytes());
    }
    assert_eq!(upstream.seen.lock().unwrap().len(), INFERENCE.len());
}

#[tokio::test]
async fn requests_outside_the_inference_routes_get_not_found_without_reaching_vllm() {
    let upstream = Upstream::start().await;
    let address = filter(upstream.address, HEADER_TIMEOUT).await;
    let absolute = format!("http://{address}/invocations");
    let absolute_allowed = format!("http://{address}/v1/models");
    // Unguarded and management routes, methods outside the allowlist, and
    // spellings that a normalizing proxy or vLLM's router could resolve to
    // another route.
    for (method, target) in [
        ("POST", "/invocations"),
        ("POST", "/tokenize"),
        ("GET", "/metrics"),
        ("POST", "/v1/load_lora_adapter"),
        ("GET", "/v1/models/fixture-model"),
        ("GET", "/version"),
        ("OPTIONS", "/v1/chat/completions"),
        ("HEAD", "/health"),
        ("POST", "//invocations"),
        ("POST", "/v1/../invocations"),
        ("POST", "/v1/%2e%2e/invocations"),
        ("POST", "/./invocations"),
        ("POST", "/health/../invocations"),
        ("POST", "/v1/..%2finvocations"),
        ("POST", "/V1/chat/completions"),
        ("POST", absolute.as_str()),
        ("GET", absolute_allowed.as_str()),
        ("POST", "/invocations?x=/v1/"),
        ("POST", "/v1invocations"),
        ("POST", "/v1/chat/completions/../../invocations"),
        ("POST", "//v1/chat/completions"),
    ] {
        let response = exchange(address, &request(method, target)).await;
        if method == "HEAD" {
            // A HEAD response has no body.
            assert_eq!(statuses(&response), ["404"], "{response}");
        } else {
            assert_not_found(&response, &format!("{method} {target}"));
        }
    }
    assert_eq!(upstream.connections.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn upgrades_and_tunnels_are_refused_without_reaching_vllm() {
    let upstream = Upstream::start().await;
    let address = filter(upstream.address, HEADER_TIMEOUT).await;
    for probe in [
        format!("GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: {KEY}\r\nConnection: Upgrade\r\nUpgrade: websocket\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n"),
        "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: Upgrade, HTTP2-Settings\r\nUpgrade: h2c\r\nHTTP2-Settings: AAMAAABkAAQAoAAAAAIAAAAA\r\n\r\n".into(),
        format!("CONNECT {address} HTTP/1.1\r\nHost: {address}\r\n\r\n"),
        "OPTIONS * HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n".into(),
    ] {
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream.write_all(probe.as_bytes()).await.unwrap();
        assert_not_found(&read_response(&mut stream, NOT_FOUND).await, &probe);
    }
    // The filter speaks only HTTP/1.1; an HTTP/2 preface closes the connection.
    let response = exchange(address, "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n").await;
    assert_eq!(response, "");
    assert_eq!(upstream.connections.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn hop_by_hop_headers_are_not_forwarded_in_either_direction() {
    let upstream = Upstream::start().await;
    let address = filter(upstream.address, HEADER_TIMEOUT).await;
    let response = exchange(
        address,
        &format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: {KEY}\r\nConnection: close, X-Client-Hop\r\nX-Client-Hop: 1\r\nKeep-Alive: timeout=5\r\nProxy-Authorization: Basic Zm9v\r\nProxy-Connection: keep-alive\r\nTE: trailers\r\nTrailer: X-Checksum\r\nX-End-To-End: kept\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n"
        ),
    )
    .await;
    assert_eq!(statuses(&response), ["200"], "{response}");
    let seen = upstream.seen.lock().unwrap();
    let request = &seen[0];
    assert_eq!(request.body, b"hello");
    assert_eq!(request.headers["x-end-to-end"], "kept");
    assert_eq!(request.headers["authorization"], KEY);
    for name in [
        "connection",
        "x-client-hop",
        "keep-alive",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
    ] {
        assert!(!request.headers.contains_key(name), "{name} reached vLLM");
    }
    let names = header_names(&response);
    assert!(names.contains(&"x-upstream".into()), "{response}");
    for name in [
        "x-upstream-hop",
        "keep-alive",
        "proxy-authenticate",
        "trailer",
        "upgrade",
    ] {
        assert!(!names.contains(&name.into()), "{name} reached the client");
    }
}

#[tokio::test]
async fn vllm_records_the_connecting_address_instead_of_forwarding_headers_from_the_client() {
    // vLLM trusts forwarding headers from loopback, where every filtered
    // request now comes from, so a client must not choose its recorded
    // address or scheme.
    let upstream = Upstream::start().await;
    let address = filter(upstream.address, HEADER_TIMEOUT).await;
    let response = exchange(
        address,
        "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Forwarded-For: 203.0.113.7\r\nX-Forwarded-For: 198.51.100.1\r\nX-Forwarded-Proto: https\r\nConnection: close\r\n\r\n",
    )
    .await;
    assert_eq!(statuses(&response), ["200"], "{response}");
    let seen = upstream.seen.lock().unwrap();
    let forwarded: Vec<_> = seen[0].headers.get_all("x-forwarded-for").iter().collect();
    assert_eq!(forwarded, ["127.0.0.1"]);
    assert!(!seen[0].headers.contains_key("x-forwarded-proto"));
}

/// Accept one connection as vLLM and read its request. Then, if `stream`,
/// send the first event of a stream, and wait for `release` before finishing
/// it. Returns whether the filter closed the connection before release.
async fn respond_once(
    listener: TcpListener,
    stream: bool,
    received: tokio::sync::oneshot::Sender<()>,
    release: tokio::sync::oneshot::Receiver<()>,
) -> bool {
    let (mut connection, _) = listener.accept().await.unwrap();
    let mut head = Vec::new();
    while !head.ends_with(b"\r\n\r\n") {
        head.push(connection.read_u8().await.unwrap());
    }
    let head = String::from_utf8(head).unwrap().to_ascii_lowercase();
    let size: usize = head
        .split("content-length: ")
        .nth(1)
        .and_then(|rest| rest.split("\r\n").next())
        .unwrap()
        .parse()
        .unwrap();
    let mut body = vec![0; size];
    connection.read_exact(&mut body).await.unwrap();
    let _ = received.send(());
    if stream {
        connection
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\nd\r\ndata: first\n\n\r\n")
            .await
            .unwrap();
    }
    let mut closed = [0; 1];
    tokio::select! {
        read = connection.read(&mut closed) => return matches!(read, Ok(0) | Err(_)),
        _ = release => {}
    }
    connection
        .write_all(b"e\r\ndata: second\n\n\r\n0\r\n\r\n")
        .await
        .unwrap();
    false
}

/// Read until the first streamed event arrives.
async fn first_event(client: &mut TcpStream) -> String {
    let mut received = Vec::new();
    tokio::time::timeout(Duration::from_secs(5), async {
        while !String::from_utf8_lossy(&received).contains("data: first") {
            let mut chunk = [0; 256];
            let read = client.read(&mut chunk).await.unwrap();
            assert_ne!(read, 0, "stream ended early");
            received.extend_from_slice(&chunk[..read]);
        }
    })
    .await
    .expect("the first event arrives while vLLM is still streaming");
    String::from_utf8(received).unwrap()
}

#[tokio::test]
async fn streamed_events_reach_the_client_before_vllm_finishes() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = filter(listener.local_addr().unwrap(), HEADER_TIMEOUT).await;
    let (received, _) = tokio::sync::oneshot::channel();
    let (release, released) = tokio::sync::oneshot::channel();
    let upstream = tokio::spawn(respond_once(listener, true, received, released));
    let mut client = TcpStream::connect(address).await.unwrap();
    client
        .write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: {KEY}\r\nContent-Length: 15\r\nConnection: close\r\n\r\n{{\"stream\":true}}").as_bytes())
        .await
        .unwrap();
    let first = first_event(&mut client).await;
    assert!(!first.contains("data: second"));
    release.send(()).unwrap();
    let mut rest = String::new();
    client.read_to_string(&mut rest).await.unwrap();
    assert!(rest.contains("data: second"), "{rest}");
    assert!(!upstream.await.unwrap());
}

#[tokio::test]
async fn a_client_that_disconnects_closes_its_vllm_request() {
    // vLLM aborts a request when its client disconnects, whether it is still
    // reading the prompt or already streaming the reply.
    for stream in [false, true] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = filter(listener.local_addr().unwrap(), HEADER_TIMEOUT).await;
        let (received_tx, received) = tokio::sync::oneshot::channel();
        let (_release, released) = tokio::sync::oneshot::channel();
        let upstream = tokio::spawn(respond_once(listener, stream, received_tx, released));
        let mut client = TcpStream::connect(address).await.unwrap();
        client
            .write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: {KEY}\r\nContent-Length: 15\r\n\r\n{{\"stream\":true}}").as_bytes())
            .await
            .unwrap();
        received.await.unwrap();
        if stream {
            first_event(&mut client).await;
        }
        drop(client);
        let closed = tokio::time::timeout(Duration::from_secs(5), upstream)
            .await
            .unwrap_or_else(|_| panic!("the filter keeps the vLLM request open; stream: {stream}"))
            .unwrap();
        assert!(closed, "stream: {stream}");
    }
}

#[tokio::test]
async fn unreachable_vllm_returns_bad_gateway() {
    let refused = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let backend = refused.local_addr().unwrap();
    drop(refused);
    let address = filter(backend, HEADER_TIMEOUT).await;
    let response = exchange(address, &request("GET", "/v1/models")).await;
    assert_eq!(statuses(&response), ["502"], "{response}");
    assert!(
        response.contains("content-type: application/json\r\n"),
        "{response}"
    );
    assert!(
        response.ends_with(r#"{"error":"Bad gateway"}"#),
        "{response}"
    );
}

#[tokio::test]
async fn a_request_body_that_cannot_be_read_gets_bad_request() {
    let upstream = Upstream::start().await;
    let address = filter(upstream.address, HEADER_TIMEOUT).await;
    // A chunk size that is not hexadecimal, and one that overflows.
    for chunks in [
        "zz\r\nabc\r\n0\r\n\r\n",
        "10000000000000003\r\nabc\r\n0\r\n\r\n",
    ] {
        let response = exchange(
            address,
            &format!("POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: {KEY}\r\nTransfer-Encoding: chunked\r\n\r\n{chunks}"),
        )
        .await;
        assert_eq!(statuses(&response), ["400"], "{chunks:?}: {response}");
        assert!(
            response.contains("content-type: application/json\r\n"),
            "{response}"
        );
        assert!(
            response.ends_with(r#"{"error":"Bad request"}"#),
            "{response}"
        );
    }
    assert!(upstream.requests().is_empty());
}

#[tokio::test]
async fn pipelined_requests_on_one_connection_are_each_filtered() {
    let upstream = Upstream::start().await;
    let address = filter(upstream.address, HEADER_TIMEOUT).await;
    // The second completion's body is a complete request; it stays a body.
    let smuggled = "POST /invocations HTTP/1.1\r\nHost: x\r\n\r\n";
    let response = exchange(
        address,
        &format!(
            "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n\
             POST /invocations HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 0\r\n\r\n\
             POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: {KEY}\r\nContent-Length: 5\r\n\r\nabcde\
             POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: {KEY}\r\nContent-Length: {}\r\n\r\n{smuggled}\
             GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: {KEY}\r\nConnection: close\r\n\r\n",
            smuggled.len()
        ),
    )
    .await;
    assert_eq!(
        statuses(&response),
        ["200", "404", "200", "200", "200"],
        "{response}"
    );
    assert_eq!(
        upstream.requests(),
        [
            "GET /health",
            "POST /v1/completions",
            "POST /v1/completions",
            "GET /v1/models"
        ]
    );
    let seen = upstream.seen.lock().unwrap();
    assert_eq!(seen[1].body, b"abcde");
    assert_eq!(seen[2].body, smuggled.as_bytes());
}

#[tokio::test]
async fn ambiguous_request_framing_never_reaches_vllm() {
    let upstream = Upstream::start().await;
    let address = filter(upstream.address, HEADER_TIMEOUT).await;
    // Transfer-Encoding overrides Content-Length; hyper drops the length and
    // closes the connection after the response, so the trailing request,
    // which a length-framing server would read, is never parsed.
    let response = exchange(
        address,
        &format!("POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: {KEY}\r\nContent-Length: 4\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\nPOST /invocations HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 0\r\n\r\n"),
    )
    .await;
    assert_eq!(statuses(&response), ["200"], "{response}");
    assert_eq!(upstream.requests(), ["POST /v1/completions"]);
    assert!(
        !upstream.seen.lock().unwrap()[0]
            .headers
            .contains_key("content-length")
    );
    // hyper rejects framing it cannot interpret unambiguously.
    for framing in [
        "Content-Length: 4\r\nTransfer-Encoding: gzip",
        "Content-Length: 4\r\nContent-Length: 5",
    ] {
        let response = exchange(
            address,
            &format!("POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: {KEY}\r\n{framing}\r\n\r\nabcde"),
        )
        .await;
        assert_eq!(statuses(&response), ["400"], "{framing}: {response}");
    }
    let response = exchange(
        address,
        "POST /v1/completions HTTP/1.0\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
    )
    .await;
    assert!(
        response.lines().next().unwrap().contains(" 400 "),
        "{response}"
    );
    assert_eq!(upstream.connections.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn incomplete_request_headers_time_out() {
    let upstream = Upstream::start().await;
    let address = filter(upstream.address, Duration::from_millis(200)).await;
    // A partial request head, and a connection that sends nothing at all.
    for partial in ["GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n", ""] {
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream.write_all(partial.as_bytes()).await.unwrap();
        let mut rest = Vec::new();
        let _ = tokio::time::timeout(Duration::from_secs(3), stream.read_to_end(&mut rest))
            .await
            .unwrap_or_else(|_| panic!("the filter keeps {partial:?} open"));
        assert!(statuses(&String::from_utf8_lossy(&rest)).is_empty());
    }
    assert_eq!(upstream.connections.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn connections_beyond_the_limit_wait_until_a_served_one_closes() {
    let upstream = Upstream::start().await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(serve(listener, upstream.address, HEADER_TIMEOUT, 2));
    let mut served = vec![keep_alive(address).await, keep_alive(address).await];
    let mut waiting = TcpStream::connect(address).await.unwrap();
    waiting
        .write_all(request("GET", "/health").as_bytes())
        .await
        .unwrap();
    let mut first = [0; 1];
    assert!(
        tokio::time::timeout(Duration::from_millis(300), waiting.read(&mut first))
            .await
            .is_err(),
        "the filter served a connection beyond its limit"
    );
    drop(served.pop());
    let mut response = Vec::new();
    tokio::time::timeout(Duration::from_secs(5), waiting.read_to_end(&mut response))
        .await
        .expect("the waiting connection is served once another closes")
        .unwrap();
    assert_eq!(statuses(&String::from_utf8(response).unwrap()), ["200"]);
    assert_eq!(upstream.connections.load(Ordering::SeqCst), 3);
}

#[test]
fn the_filter_raises_the_open_file_limit_and_leaves_descriptors_for_the_supervisor() {
    use rustix::process::{Resource, Rlimit, getrlimit, setrlimit};
    let original = getrlimit(Resource::Nofile);
    // Container runtimes commonly start processes with a soft limit of 1024.
    let soft = original.maximum.map_or(1024, |hard| hard.min(1024));
    setrlimit(
        Resource::Nofile,
        Rlimit {
            current: Some(soft),
            ..original
        },
    )
    .unwrap();
    let served = connections();
    let raised = getrlimit(Resource::Nofile);
    setrlimit(Resource::Nofile, original).unwrap();
    assert_eq!(raised.current, original.maximum);
    assert_eq!(served, capacity(original.maximum.unwrap_or(u64::MAX)));
    assert_eq!(capacity(1024), CONNECTIONS);
    // Each connection holds the client's socket and one to vLLM. As many as
    // fit beside the reserve are served, up to the cap, and at least one.
    for descriptors in [0, 129, 130, 131, 640, 641, 1024, 1 << 20, u64::MAX] {
        let served = capacity(descriptors) as u64;
        assert!((1..=CONNECTIONS as u64).contains(&served), "{descriptors}");
        if descriptors >= RESERVED_DESCRIPTORS + 2 {
            assert!(
                2 * served + RESERVED_DESCRIPTORS <= descriptors,
                "{descriptors}"
            );
        }
        assert!(
            served == CONNECTIONS as u64 || 2 * (served + 1) + RESERVED_DESCRIPTORS > descriptors,
            "{descriptors}"
        );
    }
}

#[tokio::test]
async fn binding_an_occupied_published_port_fails() {
    let occupied = TcpListener::bind("0.0.0.0:0").await.unwrap();
    assert!(bind(occupied.local_addr().unwrap().port()).await.is_err());
    // A released port may go to another test at once, so check a fresh one.
    let listener = bind(0).await.unwrap();
    assert!(listener.local_addr().unwrap().ip().is_unspecified());
}

fn sleeper() -> Box<dyn process_wrap::tokio::ChildWrapper> {
    let mut command = CommandWrap::with_new("/bin/sleep", |command| {
        command.arg("100");
    });
    command.wrap(KillOnDrop).wrap(ProcessGroup::leader());
    command.spawn().unwrap()
}

fn policy() -> supervisor::Policy {
    supervisor::Policy {
        startup_timeout: Duration::from_secs(1800),
        protection: ProtectionPolicy::new(8 * GIB, 3 * GIB, 12 * GIB, 5).unwrap(),
    }
}

async fn failed_filter() -> Infallible {
    panic!("fixture route filter failure")
}

#[tokio::test]
async fn a_failed_filter_stops_vllm_with_an_error() {
    let mut child = sleeper();
    let pid = child.id().unwrap();
    let (_samples_tx, samples) = tokio::sync::mpsc::channel(1);
    let (_ready_tx, ready) = tokio::sync::mpsc::channel(1);
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        supervise(
            policy(),
            child.as_mut(),
            Monitors {
                samples,
                ready,
                trip: CancellationToken::new(),
                report: &|_, _, _| Ok(()),
            },
            &CancellationToken::new(),
            tokio::spawn(failed_filter()),
        ),
    )
    .await
    .unwrap();
    let error = result.unwrap_err().to_string();
    assert!(error.contains("route filter"), "{error}");
    assert!(!std::path::Path::new(&format!("/proc/{pid}")).exists());
}

#[tokio::test]
async fn stopping_supervision_stops_the_filter() {
    let upstream = Upstream::start().await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let filter = tokio::spawn(serve(listener, upstream.address, HEADER_TIMEOUT, 1));
    // A served keep-alive connection stays open, idle, until the filter stops,
    // and holds the only slot, so the next connection waits in the listener's
    // backlog. Closing the listener resets it; a released port could be
    // reused by another test, so the test does not connect to it again.
    let mut idle = keep_alive(address).await;
    let mut queued = TcpStream::connect(address).await.unwrap();
    let mut child = sleeper();
    let (_samples_tx, samples) = tokio::sync::mpsc::channel(1);
    let (_ready_tx, ready) = tokio::sync::mpsc::channel(1);
    let cancel = CancellationToken::new();
    cancel.cancel();
    let result = supervise(
        policy(),
        child.as_mut(),
        Monitors {
            samples,
            ready,
            trip: CancellationToken::new(),
            report: &|_, _, _| Ok(()),
        },
        &cancel,
        filter,
    )
    .await;
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("stopped by operator")
    );
    let mut rest = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(3), idle.read_to_end(&mut rest))
        .await
        .expect("the filter closes the connections it accepted");
    let _ = tokio::time::timeout(Duration::from_secs(3), queued.read_to_end(&mut rest))
        .await
        .expect("the filter closes its listener");
}
