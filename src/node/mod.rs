//! Digital Logic Components and Gates for simulation.
//! 
//! This module defines various digital logic components such as gates (AND, OR, NOT, etc.),
//! multiplexers, demultiplexers, decoders, splitters, and registers as well as the
//! traits and structures needed to represent and simulate their behavior.
//! 
//! ## The node module notably consists of:
//! - **[`Component`]**: An interface for all digital logic components, defining methods for port configuration, initialization, and execution.
//! - **[`PortType`] and [`PortProperties`]**: Enumerations and structures to define the types and properties of ports for components.
//! - **[`PortUpdate`]**: A structure representing updates to port values during simulation.
//! - **Digital Logic Components**: Implementations of basic logic components used to simulate digital circuits.
use crate::bitarray::{BitArray, BitState};

pub use gates::*;
pub use memory::*;
pub use muxes::*;
pub use wiring::*;

mod gates;
mod memory;
mod muxes;
mod wiring;

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash, Debug)]

/// The type of ports available for a digital logic component.
pub enum PortType {
    /// A port which accepts incoming signals.
    Input, 
    /// A port which provides outgoing signals.
    Output, 
    /// A port which can accept and provide signals.
    Inout 
}
impl PortType {
    /// Checks if the port type accepts input signals.
    pub fn accepts_input(self) -> bool {
        matches!(self, PortType::Input | PortType::Inout)
    }

    /// Checks if the port type provides outgoing signals.
    pub fn accepts_output(self) -> bool {
        matches!(self, PortType::Output | PortType::Inout)
    }
}
/// The properties of a port for a digital logic component.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash, Debug)]
pub struct PortProperties {
    /// Type of the port.
    pub ty: PortType,
    /// Size of the data the port works with in bits.
    pub bitsize: u8
}

/// A struct representing an update to a port's value during simulation.
/// 
/// This struct should only be used to represent the update 
/// of a [`Output`] or [`Inout`] port.
/// 
/// [`Output`]: PortType::Output
/// [`Inout`]: PortType::Inout
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct PortUpdate {
    /// Index of the port being updated.
    /// 
    /// For example, this is an index of 0 when referring
    /// to the first port in a function's port list.
    pub index: usize,
    /// The new value to be assigned to the port at the given index.
    pub value: BitArray
}

/// The interface defining how a digital logic component operates.
pub trait Component {
    /// Returns the vector holding the properties of all ports associated with the component.
    /// 
    /// This is called only once during initialization.
    /// It is assumed that the result of this function will not change when called multiple times.
    fn ports(&self) -> Vec<PortProperties>;
    
    /// Initializes the state (e.g., internal state and port state) of the component.
    /// 
    /// If not specified, by default, the initial port state is set to all floating.
    fn initialize(&self, _state: &mut [BitArray]) {}
    
    /// "Runs" the component's function on a set of inputs, outputting a vector of updated ports
    /// after the function is applied.
    /// 
    /// This function is called after an update is propagated to this component.
    /// When that occurs, this function is called with the original state and updated state
    /// of this component's ports.
    /// 
    /// This function may also panic if `old_inp` and `inp` do not match the port properties
    /// specified by [`Component::ports`].
    #[must_use]
    fn run(&self, old_ports: &[BitArray], new_ports: &[BitArray]) -> Vec<PortUpdate>{
        self.validate_ports(old_ports);
        self.validate_ports(new_ports);
        self.run_inner(old_ports, new_ports)
    }

    /// Inner run function that, given a set of inputs, applies its modifications to output a vector
    /// of updated ports. This function is wrapped by run to ensure input validation
    fn run_inner(&self, old_ports: &[BitArray], new_ports: &[BitArray]) -> Vec<PortUpdate>;

    /// Validates inputs to ensure all ports match port bitsize.
    fn validate_ports(&self, ports: &[BitArray]) {
        // Only run in debug mode
        if cfg!(debug_assertions) {
            let port_props = self.ports();
            for (i, (bit_vec, port)) in ports.iter().zip(port_props).enumerate() {
                debug_assert_eq!(
                    bit_vec.len(),
                    port.bitsize,
                    "Port {i} has incorrect bit width"
                );
            }
        }
    }
}


/// The triggering conditions for components based on a signal change.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Sensitivity {
    /// Triggered on any change in a signal (rising or falling edge).
    Anyedge, 
    /// Triggered on rising edge of a signal (low to high clock transition).
    Posedge, 
    /// Triggered on falling edge of a signal (high to low clock transition).
    Negedge, 
    /// Does not update in response to a signal update.
    DontCare
}
impl Sensitivity {
    /// Checks whether the change between the old and new value
    /// would create a trigger based on this sensitivity.
    /// 
    /// ```
    /// use circuitsim_engine::bitarray::bitarr;
    /// use circuitsim_engine::node::Sensitivity;
    /// 
    /// let lo = bitarr![0];
    /// let hi = bitarr![1];
    /// assert!(Sensitivity::Posedge.activated(lo, hi));
    /// assert!(Sensitivity::Negedge.activated(hi, lo));
    /// ```
    pub fn activated(self, old: BitArray, new: BitArray) -> bool {
        assert_eq!(old.len(), new.len(), "Bit length should be the same");
        match self {
            Sensitivity::Anyedge  => old != new,
            Sensitivity::Posedge  => old.all(BitState::Low) && new.all(BitState::High),
            Sensitivity::Negedge  => old.all(BitState::High) && new.all(BitState::Low),
            Sensitivity::DontCare => false,
        }
    }

    /// Checks whether the changes in values (as specified by the `old` and `new` slices)
    /// would create a trigger based on this sensitivity.
    /// 
    /// ```
    /// use circuitsim_engine::bitarray::bitarr;
    /// use circuitsim_engine::node::Sensitivity;
    /// 
    /// let lo = bitarr![0];
    /// let hi = bitarr![1];
    /// assert!(Sensitivity::Posedge.any_activated(&[lo, lo, lo, lo], &[lo, hi, lo, lo]));
    /// ```
    /// 
    /// This function panics if `old.len() != new.len()`.
    pub fn any_activated(self, old: &[BitArray], new: &[BitArray]) -> bool {
        assert_eq!(old.len(), new.len(), "Array size should be the same");
        std::iter::zip(old, new)
            .any(|(&o, &n)| self.activated(o, n))
    }
}

/// Helper function to more easily define port lists (for [`Component::ports`]).
fn port_list(config: &[(PortProperties, u8)]) -> Vec<PortProperties> {
    config.iter()
        .flat_map(|&(props, ct)| std::iter::repeat_n(props, usize::from(ct)))
        .collect()
}
macro_rules! decl_component_enum {
    ($ComponentEnum:ident: $($Component:ident),*$(,)?) => {
        /// An enum that represents all supported digital logic components.
        #[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
        pub enum $ComponentEnum {
            $(
                #[allow(missing_docs)]
                $Component($Component)
            ),*
        }
        impl Component for $ComponentEnum {
            fn ports(&self) -> Vec<PortProperties> {
                match self {
                    $(
                        Self::$Component(c) => c.ports(),
                    )*
                }
            }
            fn initialize(&self, state: &mut [BitArray]) {
                match self {
                    $(
                        Self::$Component(c) => c.initialize(state),
                    )*
                }
            }
            fn run_inner(&self, old_ports: &[BitArray], new_ports: &[BitArray]) -> Vec<PortUpdate> {
                match self {
                    $(
                        Self::$Component(c) => c.run(old_ports, new_ports),
                    )*
                }
            }
        }
        $(
            impl From<$Component> for $ComponentEnum {
                fn from(value: $Component) -> Self {
                    Self::$Component(value)
                }
            }
        )*
    }
}
decl_component_enum!(ComponentFn: 
    And, Or, Xor, Nand, Nor, Xnor, Not, TriState, Input, Output, Constant,
    Mux, Demux, Decoder, Splitter, Register
);

#[cfg(test)]
fn floating_ports(properties: &[PortProperties]) -> Vec<BitArray> {
    use crate::bitarr;

    properties.iter()
        .map(|p| bitarr![Z; p.bitsize])
        .collect()
}
