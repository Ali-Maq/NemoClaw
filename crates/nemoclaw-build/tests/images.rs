// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Image labels publish exactly the metadata the image itself reports.

use nemoclaw_build::images::{PLATFORMS, labels, plan};
use serde_json::json;
use std::path::Path;

#[test]
fn harness_images_carry_a_catalog_that_agrees_with_their_bridge() {
    let bridge = json!({"interface_version": 1});
    let catalog = json!({"bridge": bridge, "adapters": []});
    let published = labels("openclaw", bridge.clone(), catalog.clone()).unwrap();
    assert_eq!(
        published,
        [
            ("io.nemoclaw.fabric.bridge".to_owned(), bridge.to_string()),
            ("io.nemoclaw.fabric.catalog".to_owned(), catalog.to_string()),
        ]
    );
    let disagreeing = json!({"bridge": {"interface_version": 2}});
    assert!(labels("openclaw", bridge.clone(), disagreeing).is_err());
    // Only the reference image may publish the reference adapter.
    let impostor = json!({"bridge": bridge, "adapters": [{"descriptor": {"adapter_id": "org.nemoclaw.dummy"}}]});
    assert!(labels("openclaw", bridge, impostor).is_err());
}

#[test]
fn the_reference_image_carries_a_catalog_so_deployments_can_select_it() {
    let bridge = json!({"interface_version": 1});
    let catalog = json!({"bridge": bridge, "adapters": [{"descriptor": {"adapter_id": "org.nemoclaw.dummy"}}]});
    assert_eq!(
        labels("dummy", bridge.clone(), catalog.clone()).unwrap(),
        [
            ("io.nemoclaw.fabric.bridge".to_owned(), bridge.to_string()),
            ("io.nemoclaw.fabric.catalog".to_owned(), catalog.to_string()),
        ]
    );
    // A catalog from Fabric discovery, read by mistake, lists other adapters.
    let discovered =
        json!({"bridge": bridge, "adapters": [{"descriptor": {"adapter_id": "nvidia.fabric.pi"}}]});
    assert!(labels("dummy", bridge.clone(), discovered).is_err());
    assert!(
        labels(
            "dummy",
            bridge.clone(),
            json!({"bridge": bridge, "adapters": []})
        )
        .is_err()
    );
}

#[test]
fn builds_require_one_of_the_qualified_platforms() {
    assert_eq!(PLATFORMS, ["linux/arm64", "linux/amd64"]);
    // Rejected before Docker runs, so this needs no daemon.
    let error = plan(Path::new("."), "linux/riscv64", &["agents".into()]).unwrap_err();
    assert!(error.contains("--platform"), "{error}");
}

#[test]
fn harnesses_missing_from_the_selected_platform_name_the_platforms_that_build_them() {
    // Plans with Bake's print mode only; nothing is built.
    if !std::process::Command::new("docker")
        .args(["buildx", "version"])
        .output()
        .is_ok_and(|output| output.status.success())
    {
        eprintln!("skipping: docker buildx is unavailable");
        return;
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let error = plan(&root, "linux/amd64", &["openclaw".into(), "hermes".into()]).unwrap_err();
    assert_eq!(
        error,
        "hermes is built only for linux/arm64; linux/amd64 builds deepagents, openclaw (docs/build.md#build-agent-images)"
    );
    let planned = |platform, name: &str| {
        plan(&root, platform, &[name.into()])
            .unwrap()
            .into_iter()
            .map(|(target, _)| target)
            .collect::<Vec<_>>()
    };
    assert_eq!(planned("linux/amd64", "openclaw"), ["openclaw"]);
    assert_eq!(planned("linux/arm64", "hermes"), ["hermes"]);
    // Selections that are not harnesses keep Bake's own resolution.
    assert_eq!(planned("linux/amd64", "dummy"), ["dummy"]);
    assert!(plan(&root, "linux/amd64", &["no-such-target".into()]).is_err());
}
