//! Tests for middle-end representation operations

#[cfg(test)]
mod operations_tests {
    use super::super::*;

    #[test]
    fn test_component_position_operations() {
        let mut mc = MiddleCircuit::new();
        
        // Add component
        let pos = PositionData { x: 10, y: 20, properties: None };
        let key = mc.add_component(crate::func::Input::new(1), pos.clone());
        
        // Get position
        assert_eq!(mc.get_component_position(key).unwrap().x, 10);
        assert_eq!(mc.get_component_position(key).unwrap().y, 20);
        
        // Update position
        let new_pos = PositionData { x: 30, y: 40, properties: None };
        assert!(mc.update_component_position(key, new_pos));
        assert_eq!(mc.get_component_position(key).unwrap().x, 30);
        
        // Check existence
        assert!(mc.has_component(key));
    }

    #[test]
    fn test_component_count() {
        let mut mc = MiddleCircuit::new();
        
        assert_eq!(mc.component_count(), 0);
        
        mc.add_component(
            crate::func::Input::new(1),
            PositionData { x: 0, y: 0, properties: None }
        );
        assert_eq!(mc.component_count(), 1);
        
        mc.add_component(
            crate::func::Output::new(1),
            PositionData { x: 10, y: 0, properties: None }
        );
        assert_eq!(mc.component_count(), 2);
    }

    #[test]
    fn test_move_component() {
        let mut mc = MiddleCircuit::new();
        
        let key = mc.add_component(
            crate::func::Input::new(1),
            PositionData { x: 10, y: 20, properties: None }
        );
        
        // Move by delta
        assert!(mc.move_component(key, 5, -3));
        
        let pos = mc.get_component_position(key).unwrap();
        assert_eq!(pos.x, 15);
        assert_eq!(pos.y, 17);
    }

    #[test]
    fn test_move_all_components() {
        let mut mc = MiddleCircuit::new();
        
        let key1 = mc.add_component(
            crate::func::Input::new(1),
            PositionData { x: 10, y: 20, properties: None }
        );
        let key2 = mc.add_component(
            crate::func::Output::new(1),
            PositionData { x: 30, y: 40, properties: None }
        );
        
        mc.move_all_components(5, -10);
        
        assert_eq!(mc.get_component_position(key1).unwrap().x, 15);
        assert_eq!(mc.get_component_position(key1).unwrap().y, 10);
        assert_eq!(mc.get_component_position(key2).unwrap().x, 35);
        assert_eq!(mc.get_component_position(key2).unwrap().y, 30);
    }

    #[test]
    fn test_find_components_in_region() {
        let mut mc = MiddleCircuit::new();
        
        mc.add_component(
            crate::func::Input::new(1),
            PositionData { x: 10, y: 10, properties: None }
        );
        mc.add_component(
            crate::func::Output::new(1),
            PositionData { x: 50, y: 50, properties: None }
        );
        mc.add_component(
            crate::func::And::new(1, 2),
            PositionData { x: 30, y: 30, properties: None }
        );
        
        // Find components in region (0,0) to (40,40)
        let found = mc.find_components_in_region(0, 0, 40, 40);
        assert_eq!(found.len(), 2); // Should find input and AND gate
        
        // Find in smaller region
        let found = mc.find_components_in_region(0, 0, 20, 20);
        assert_eq!(found.len(), 1); // Should only find input
    }

    #[test]
    fn test_clone_component() {
        let mut mc = MiddleCircuit::new();
        
        let original_key = mc.add_component(
            crate::func::And::new(8, 2),
            PositionData { x: 10, y: 10, properties: None }
        );
        
        let new_pos = PositionData { x: 50, y: 50, properties: None };
        let cloned_key = mc.clone_component(original_key, new_pos).unwrap();
        
        // Should have different keys
        assert_ne!(original_key, cloned_key);
        
        // Should have 2 components now
        assert_eq!(mc.component_count(), 2);
        
        // Check positions
        assert_eq!(mc.get_component_position(original_key).unwrap().x, 10);
        assert_eq!(mc.get_component_position(cloned_key).unwrap().x, 50);
    }

    #[test]
    fn test_component_properties() {
        let mut mc = MiddleCircuit::new();
        
        let key = mc.add_component(
            crate::func::Input::new(1),
            PositionData { 
                x: 10, 
                y: 10, 
                properties: Some([("Label".to_string(), "Input1".to_string())].into())
            }
        );
        
        // Get property
        assert_eq!(mc.get_component_property(key, "Label").unwrap(), "Input1");
        
        // Set property
        assert!(mc.set_component_property(key, "Direction".to_string(), "EAST".to_string()));
        assert_eq!(mc.get_component_property(key, "Direction").unwrap(), "EAST");
        
        // Remove property
        let removed = mc.remove_component_property(key, "Label");
        assert_eq!(removed.unwrap(), "Input1");
        assert!(mc.get_component_property(key, "Label").is_none());
    }

    #[test]
    fn test_wire_operations() {
        let mut mc = MiddleCircuit::new();
        let value_key = mc.engine.add_value_node();
        
        assert_eq!(mc.wire_count(), 0);
        
        // Add wire mesh
        let mesh1 = WireMesh { x: 10, y: 10, length: 20, is_horizontal: true };
        mc.add_wire_mesh(value_key, mesh1);
        
        assert_eq!(mc.wire_count(), 1);
        assert!(mc.has_wire(value_key));
        
        // Get wire meshes
        let meshes = mc.get_wire_meshes(value_key).unwrap();
        assert_eq!(meshes.len(), 1);
        assert_eq!(meshes[0].x, 10);
        
        // Add another mesh to same wire
        let mesh2 = WireMesh { x: 30, y: 10, length: 15, is_horizontal: false };
        mc.add_wire_mesh(value_key, mesh2);
        
        let meshes = mc.get_wire_meshes(value_key).unwrap();
        assert_eq!(meshes.len(), 2);
        
        // Remove wire meshes
        let removed = mc.remove_wire_meshes(value_key);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().len(), 2);
        assert_eq!(mc.wire_count(), 0);
    }

    #[test]
    fn test_connect_components() {
        let mut mc = MiddleCircuit::new();
        
        let input_key = mc.add_component(
            crate::func::Input::new(1),
            PositionData { x: 10, y: 10, properties: None }
        );
        
        let output_key = mc.add_component(
            crate::func::Output::new(1),
            PositionData { x: 50, y: 10, properties: None }
        );
        
        // Connect input port 0 to output port 0
        let wire = mc.connect_components(input_key, 0, output_key, 0);
        
        // Wire should exist in engine
        assert!(mc.engine.graph().values.contains_key(wire));
    }

    #[test]
    fn test_extras_operations() {
        let mut mc = MiddleCircuit::new();
        
        assert_eq!(mc.extras().len(), 0);
        
        // Add extras
        mc.add_extra(serde_json::json!({
            "type": "Probe",
            "x": 10,
            "y": 20
        }));
        
        mc.add_extra(serde_json::json!({
            "type": "Text",
            "text": "Hello",
            "x": 30,
            "y": 40
        }));
        
        assert_eq!(mc.extras().len(), 2);
        
        // Clear extras
        mc.clear_extras();
        assert_eq!(mc.extras().len(), 0);
    }

    #[test]
    fn test_middle_end_operations() {
        let mut me = MiddleEnd::new();
        
        assert_eq!(me.circuit_count(), 0);
        
        // Add circuits
        me.add_circuit("Circuit1".to_string(), MiddleCircuit::new());
        me.add_circuit("Circuit2".to_string(), MiddleCircuit::new());
        
        assert_eq!(me.circuit_count(), 2);
        assert!(me.has_circuit("Circuit1"));
        assert!(me.has_circuit("Circuit2"));
        
        // Get circuit names
        let names = me.circuit_names();
        assert_eq!(names.len(), 2);
        
        // Get circuit
        assert!(me.get_circuit("Circuit1").is_some());
        assert!(me.get_circuit("NonExistent").is_none());
    }

    #[test]
    fn test_rename_circuit() {
        let mut me = MiddleEnd::new();
        
        me.add_circuit("OldName".to_string(), MiddleCircuit::new());
        
        assert!(me.rename_circuit("OldName", "NewName".to_string()));
        
        assert!(!me.has_circuit("OldName"));
        assert!(me.has_circuit("NewName"));
    }

    #[test]
    fn test_clone_circuit() {
        let mut me = MiddleEnd::new();
        
        // Create a circuit with components
        let mut circuit = MiddleCircuit::new();
        circuit.add_component(
            crate::func::Input::new(8),
            PositionData { x: 10, y: 10, properties: None }
        );
        circuit.add_component(
            crate::func::Output::new(8),
            PositionData { x: 50, y: 10, properties: None }
        );
        
        me.add_circuit("Original".to_string(), circuit);
        
        // Clone it
        assert!(me.clone_circuit("Original", "Clone".to_string()));
        
        assert_eq!(me.circuit_count(), 2);
        
        // Both should have components
        let original = me.get_circuit("Original").unwrap();
        let clone = me.get_circuit("Clone").unwrap();
        
        assert_eq!(original.component_count(), 2);
        assert_eq!(clone.component_count(), 2);
    }

    #[test]
    fn test_remove_circuit() {
        let mut me = MiddleEnd::new();
        
        me.add_circuit("ToRemove".to_string(), MiddleCircuit::new());
        me.add_circuit("ToKeep".to_string(), MiddleCircuit::new());
        
        assert_eq!(me.circuit_count(), 2);
        
        let removed = me.remove_circuit("ToRemove");
        assert!(removed.is_some());
        
        assert_eq!(me.circuit_count(), 1);
        assert!(!me.has_circuit("ToRemove"));
        assert!(me.has_circuit("ToKeep"));
    }

    #[test]
    fn test_clear_all_circuits() {
        let mut me = MiddleEnd::new();
        
        me.add_circuit("Circuit1".to_string(), MiddleCircuit::new());
        me.add_circuit("Circuit2".to_string(), MiddleCircuit::new());
        me.add_circuit("Circuit3".to_string(), MiddleCircuit::new());
        
        assert_eq!(me.circuit_count(), 3);
        
        me.clear();
        
        assert_eq!(me.circuit_count(), 0);
    }

    #[test]
    fn test_middle_end_iterators() {
        let mut me = MiddleEnd::new();
        
        me.add_circuit("A".to_string(), MiddleCircuit::new());
        me.add_circuit("B".to_string(), MiddleCircuit::new());
        me.add_circuit("C".to_string(), MiddleCircuit::new());
        
        // Test immutable iterator
        let count = me.iter().count();
        assert_eq!(count, 3);
        
        // Test mutable iterator
        for (_name, circuit) in me.iter_mut() {
            circuit.add_component(
                crate::func::Input::new(1),
                PositionData { x: 0, y: 0, properties: None }
            );
        }
        
        // All circuits should have 1 component now
        for (_, circuit) in me.iter() {
            assert_eq!(circuit.component_count(), 1);
        }
    }

    #[test]
    fn test_component_positions_iterator() {
        let mut mc = MiddleCircuit::new();
        
        mc.add_component(
            crate::func::Input::new(1),
            PositionData { x: 10, y: 10, properties: None }
        );
        mc.add_component(
            crate::func::Output::new(1),
            PositionData { x: 50, y: 50, properties: None }
        );
        
        let positions: Vec<_> = mc.component_positions().collect();
        assert_eq!(positions.len(), 2);
    }

    #[test]
    fn test_wire_positions_iterator() {
        let mut mc = MiddleCircuit::new();
        
        let v1 = mc.engine.add_value_node();
        let v2 = mc.engine.add_value_node();
        
        mc.add_wire_mesh(v1, WireMesh { x: 10, y: 10, length: 20, is_horizontal: true });
        mc.add_wire_mesh(v2, WireMesh { x: 30, y: 30, length: 15, is_horizontal: false });
        
        let wire_positions: Vec<_> = mc.wire_positions().collect();
        assert_eq!(wire_positions.len(), 2);
    }

    #[test]
    fn test_disconnect_wire() {
        let mut mc = MiddleCircuit::new();
        let value_key = mc.engine.add_value_node();
        
        mc.add_wire_mesh(value_key, WireMesh { 
            x: 10, y: 10, length: 20, is_horizontal: true 
        });
        
        assert!(mc.has_wire(value_key));
        
        mc.disconnect_wire(value_key);
        
        assert!(!mc.has_wire(value_key));
    }
}
