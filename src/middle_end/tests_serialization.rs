//! Tests for middle-end serialization and deserialization.

use super::*;
use std::path::Path;

#[test]
fn test_deserialize_legacy_json_basic() {
    // Test with the actual latches.sim file
    let json_content = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/middle_end/latches.sim")
    ).expect("Failed to read latches.sim");

    let result = MiddleEnd::from_legacy_json(&json_content);
    assert!(result.is_ok(), "Failed to deserialize legacy JSON: {:?}", result.err());

    let me = result.unwrap();
    
    // Verify we have the expected circuits
    assert_eq!(me.circuits.len(), 4, "Expected 4 circuits in latches.sim");
    assert!(me.circuits.contains_key("RS Latch"));
    assert!(me.circuits.contains_key("Gated D Latch"));
    assert!(me.circuits.contains_key("D Flip-Flop"));
    assert!(me.circuits.contains_key("Register"));
}

#[test]
fn test_deserialize_rs_latch_components() {
    let json_content = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/middle_end/latches.sim")
    ).expect("Failed to read latches.sim");

    let me = MiddleEnd::from_legacy_json(&json_content).unwrap();
    let rs_latch = me.circuits.get("RS Latch").expect("RS Latch not found");

    // RS Latch should have 4 components (2 inputs, 2 outputs)
    assert_eq!(rs_latch.component_pos.len(), 4, "Expected 4 components in RS Latch");
    
    // Verify position data exists
    let positions: Vec<_> = rs_latch.component_pos.values().collect();
    assert!(positions.iter().any(|p| p.x == 36 && p.y == 19)); // NotOut
    assert!(positions.iter().any(|p| p.x == 10 && p.y == 10)); // S
    assert!(positions.iter().any(|p| p.x == 10 && p.y == 19)); // R
    assert!(positions.iter().any(|p| p.x == 36 && p.y == 10)); // Out
}

#[test]
fn test_deserialize_register_multibit() {
    let json_content = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/middle_end/latches.sim")
    ).expect("Failed to read latches.sim");

    let me = MiddleEnd::from_legacy_json(&json_content).unwrap();
    let register = me.circuits.get("Register").expect("Register not found");

    // Register should have 3 components
    assert_eq!(register.component_pos.len(), 3, "Expected 3 components in Register");

    // Check that we have different bitsizes
    let positions: Vec<_> = register.component_pos.values().collect();
    
    // Find the 4-bit input and output
    let four_bit_components = positions.iter().filter(|p| {
        p.properties.as_ref()
            .and_then(|props| props.get("Bitsize"))
            .map(|s| s == "4")
            .unwrap_or(false)
    }).count();
    
    assert_eq!(four_bit_components, 2, "Expected 2 components with 4-bit width");
}

#[test]
fn test_serialize_to_v2_json() {
    let json_content = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/middle_end/latches.sim")
    ).expect("Failed to read latches.sim");

    let me = MiddleEnd::from_legacy_json(&json_content).unwrap();
    
    let v2_json = me.to_v2_json();
    assert!(v2_json.is_ok(), "Failed to serialize to V2 JSON: {:?}", v2_json.err());

    let json_str = v2_json.unwrap();
    
    // Verify it's valid JSON
    let parsed: Result<Vec<MiddleCircuitV2>, _> = serde_json::from_str(&json_str);
    assert!(parsed.is_ok(), "V2 JSON is not valid: {:?}", parsed.err());
    
    let circuits = parsed.unwrap();
    assert_eq!(circuits.len(), 4, "Expected 4 circuits in V2 output");
}

#[test]
fn test_v2_circuit_structure() {
    let json_content = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/middle_end/latches.sim")
    ).expect("Failed to read latches.sim");

    let me = MiddleEnd::from_legacy_json(&json_content).unwrap();
    let v2_json = me.to_v2_json().unwrap();
    let circuits: Vec<MiddleCircuitV2> = serde_json::from_str(&v2_json).unwrap();
    
    // Check RS Latch structure
    let rs_latch = circuits.iter().find(|c| c.name == "RS Latch").expect("RS Latch not found");
    
    // Should have functions and values
    assert!(rs_latch.functions.len() > 0, "Expected functions in RS Latch");
    assert!(rs_latch.values.len() > 0, "Expected values in RS Latch");
    
    // Should have position data
    assert_eq!(rs_latch.component_pos.len(), 4, "Expected 4 component positions");
    
    // Verify component_pos references valid function indices
    for (func_idx, _) in &rs_latch.component_pos {
        assert!(*func_idx < rs_latch.functions.len(), "Invalid function index in component_pos");
    }
}

#[test]
fn test_function_serialization() {
    let json_content = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/middle_end/latches.sim")
    ).expect("Failed to read latches.sim");

    let me = MiddleEnd::from_legacy_json(&json_content).unwrap();
    let v2_json = me.to_v2_json().unwrap();
    let circuits: Vec<MiddleCircuitV2> = serde_json::from_str(&v2_json).unwrap();
    
    for circuit in &circuits {
        for func in &circuit.functions {
            // Verify function has a type
            assert!(!func.ty.is_empty(), "Function type should not be empty");
            
            // Verify port_bits and links have same length
            assert_eq!(func.port_bits.len(), func.links.len(), 
                "Port bits and links should have same length");
            
            // Verify linked value indices are valid
            for opt_val_idx in &func.links {
                if let Some(val_idx) = opt_val_idx {
                    assert!(*val_idx < circuit.values.len(), 
                        "Invalid value index {} in function link", val_idx);
                }
            }
        }
    }
}

#[test]
fn test_value_serialization() {
    let json_content = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/middle_end/latches.sim")
    ).expect("Failed to read latches.sim");

    let me = MiddleEnd::from_legacy_json(&json_content).unwrap();
    let v2_json = me.to_v2_json().unwrap();
    let circuits: Vec<MiddleCircuitV2> = serde_json::from_str(&v2_json).unwrap();
    
    for circuit in &circuits {
        for value in &circuit.values {
            // Verify all links reference valid function indices
            for (func_idx, port_idx) in &value.links {
                assert!(*func_idx < circuit.functions.len(), 
                    "Invalid function index {} in value link", func_idx);
                
                // Verify port index is valid for the referenced function
                let func = &circuit.functions[*func_idx];
                assert!(*port_idx < func.port_bits.len(), 
                    "Invalid port index {} for function with {} ports", 
                    port_idx, func.port_bits.len());
            }
        }
    }
}

#[test]
fn test_empty_middle_end() {
    let me = MiddleEnd::default();
    
    let v2_json = me.to_v2_json();
    assert!(v2_json.is_ok(), "Failed to serialize empty MiddleEnd");
    
    let json_str = v2_json.unwrap();
    let circuits: Vec<MiddleCircuitV2> = serde_json::from_str(&json_str).unwrap();
    assert_eq!(circuits.len(), 0, "Empty MiddleEnd should produce empty circuits array");
}

#[test]
fn test_middle_circuit_new() {
    let mc = MiddleCircuit::new();
    
    assert_eq!(mc.engine.graph().functions.len(), 0, "New circuit should have no functions");
    assert_eq!(mc.engine.graph().values.len(), 0, "New circuit should have no values");
    assert_eq!(mc.component_pos.len(), 0, "New circuit should have no component positions");
    assert_eq!(mc.wire_pos.len(), 0, "New circuit should have no wire positions");
    assert_eq!(mc.extras.len(), 0, "New circuit should have no extras");
}

#[test]
fn test_invalid_json() {
    let invalid_json = r#"{"invalid": "json"}"#;
    let result = MiddleEnd::from_legacy_json(invalid_json);
    assert!(result.is_err(), "Should fail on invalid JSON structure");
}

#[test]
fn test_malformed_version() {
    let json = r#"{
        "version": "invalid",
        "circuits": []
    }"#;
    
    // Should still deserialize since we don't validate version format
    let result = MiddleEnd::from_legacy_json(json);
    assert!(result.is_ok(), "Should handle any version string");
}

#[test]
fn test_circuit_with_no_components() {
    let json = r#"{
        "version": "1.9.1",
        "circuits": [{
            "name": "Empty",
            "components": [],
            "wires": []
        }]
    }"#;
    
    let me = MiddleEnd::from_legacy_json(json).unwrap();
    assert_eq!(me.circuits.len(), 1);
    
    let empty = me.circuits.get("Empty").unwrap();
    assert_eq!(empty.component_pos.len(), 0);
}

#[test]
fn test_roundtrip_consistency() {
    let json_content = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/middle_end/latches.sim")
    ).expect("Failed to read latches.sim");

    let me1 = MiddleEnd::from_legacy_json(&json_content).unwrap();
    let v2_json = me1.to_v2_json().unwrap();
    
    // Parse the V2 JSON to verify structure
    let circuits: Vec<MiddleCircuitV2> = serde_json::from_str(&v2_json).unwrap();
    
    // Verify all original circuit names are present
    let original_names: Vec<_> = me1.circuits.keys().collect();
    let v2_names: Vec<_> = circuits.iter().map(|c| &c.name).collect();
    
    for name in original_names {
        assert!(v2_names.contains(&name), "Circuit '{}' missing in V2 export", name);
    }
}

#[test]
fn test_wire_mesh_preservation() {
    let json = r#"{
        "version": "1.9.1",
        "circuits": [{
            "name": "WithWires",
            "components": [],
            "wires": [
                {"x": 10, "y": 20, "length": 5, "isHorizontal": true},
                {"x": 15, "y": 20, "length": 3, "isHorizontal": false}
            ]
        }]
    }"#;
    
    let me = MiddleEnd::from_legacy_json(json).unwrap();
    let circuit = me.circuits.get("WithWires").unwrap();
    
    // Wires should be stored in extras
    assert_eq!(circuit.extras.len(), 2, "Wire meshes should be preserved in extras");
}

#[test]
fn test_component_properties_preservation() {
    let json_content = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/middle_end/latches.sim")
    ).expect("Failed to read latches.sim");

    let me = MiddleEnd::from_legacy_json(&json_content).unwrap();
    let rs_latch = me.circuits.get("RS Latch").unwrap();
    
    // Check that properties are preserved
    let has_properties = rs_latch.component_pos.values().any(|pos| {
        pos.properties.is_some()
    });
    
    assert!(has_properties, "Component properties should be preserved");
    
    // Check specific property values
    let has_label = rs_latch.component_pos.values().any(|pos| {
        pos.properties.as_ref()
            .and_then(|props| props.get("Label"))
            .map(|label| !label.is_empty())
            .unwrap_or(false)
    });
    
    assert!(has_label, "Component labels should be preserved");
}

#[test]
fn test_bitsize_handling() {
    let json = r#"{
        "version": "1.9.1",
        "circuits": [{
            "name": "MultiBit",
            "components": [{
                "name": "com.ra4king.circuitsim.gui.peers.wiring.PinPeer",
                "x": 10,
                "y": 10,
                "properties": {
                    "Is input?": "Yes",
                    "Bitsize": "8"
                }
            }],
            "wires": []
        }]
    }"#;
    
    let me = MiddleEnd::from_legacy_json(json).unwrap();
    let circuit = me.circuits.get("MultiBit").unwrap();
    
    // Should create appropriate component
    assert_eq!(circuit.component_pos.len(), 1);
    
    // Verify bitsize is stored in properties
    let pos = circuit.component_pos.values().next().unwrap();
    assert_eq!(
        pos.properties.as_ref().unwrap().get("Bitsize").unwrap(),
        "8"
    );
}

#[test]
fn test_input_output_distinction() {
    let json = r#"{
        "version": "1.9.1",
        "circuits": [{
            "name": "IOTest",
            "components": [
                {
                    "name": "com.ra4king.circuitsim.gui.peers.wiring.PinPeer",
                    "x": 10,
                    "y": 10,
                    "properties": {
                        "Is input?": "Yes",
                        "Bitsize": "1"
                    }
                },
                {
                    "name": "com.ra4king.circuitsim.gui.peers.wiring.PinPeer",
                    "x": 20,
                    "y": 10,
                    "properties": {
                        "Is input?": "No",
                        "Bitsize": "1"
                    }
                }
            ],
            "wires": []
        }]
    }"#;
    
    let me = MiddleEnd::from_legacy_json(json).unwrap();
    let circuit = me.circuits.get("IOTest").unwrap();
    
    // Should have both components
    assert_eq!(circuit.component_pos.len(), 2);
}

#[test]
fn test_v2_json_is_pretty_printed() {
    let json = r#"{
        "version": "1.9.1",
        "circuits": [{
            "name": "Simple",
            "components": [],
            "wires": []
        }]
    }"#;
    
    let me = MiddleEnd::from_legacy_json(json).unwrap();
    let v2_json = me.to_v2_json().unwrap();
    
    // Pretty-printed JSON should have newlines
    assert!(v2_json.contains('\n'), "V2 JSON should be pretty-printed");
}

#[test]
fn test_multiple_circuits() {
    let json = r#"{
        "version": "1.9.1",
        "circuits": [
            {
                "name": "Circuit1",
                "components": [],
                "wires": []
            },
            {
                "name": "Circuit2",
                "components": [],
                "wires": []
            },
            {
                "name": "Circuit3",
                "components": [],
                "wires": []
            }
        ]
    }"#;
    
    let me = MiddleEnd::from_legacy_json(json).unwrap();
    assert_eq!(me.circuits.len(), 3);
    assert!(me.circuits.contains_key("Circuit1"));
    assert!(me.circuits.contains_key("Circuit2"));
    assert!(me.circuits.contains_key("Circuit3"));
}

#[cfg(test)]
mod file_io_tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn test_from_legacy_file() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/middle_end/latches.sim");
        
        let result = MiddleEnd::from_legacy_file(&path);
        assert!(result.is_ok(), "Failed to load from file: {:?}", result.err());
        
        let me = result.unwrap();
        assert_eq!(me.circuits.len(), 4);
    }

    #[test]
    fn test_write_v2_file() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/middle_end/latches.sim");
        
        let me = MiddleEnd::from_legacy_file(&path).unwrap();
        
        // Write to temp file in the target directory
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("test_output.json");
        
        let write_result = me.write_v2_file(&temp_path);
        assert!(write_result.is_ok(), "Failed to write V2 file: {:?}", write_result.err());
        
        // Verify file exists and is valid JSON
        assert!(temp_path.exists());
        let content = fs::read_to_string(&temp_path).unwrap();
        let parsed: Result<Vec<MiddleCircuitV2>, _> = serde_json::from_str(&content);
        assert!(parsed.is_ok(), "Written file is not valid JSON");
        
        // Cleanup
        let _ = fs::remove_file(&temp_path);
    }

    #[test]
    fn test_from_legacy_file_not_found() {
        let path = PathBuf::from("nonexistent_file.sim");
        let result = MiddleEnd::from_legacy_file(&path);
        assert!(result.is_err(), "Should fail on nonexistent file");
    }
}
