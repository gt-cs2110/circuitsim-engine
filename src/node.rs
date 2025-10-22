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

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash, Debug)]

/// PortType defines the type of port for a digital logic component.
/// - `Input`: Port accepts incoming signals  
/// - `Output`: Port produces outgoing signals  
/// - `Inout`: Port can both accept and provide signals
pub enum PortType {
    /// Port accepts incoming signals
    Input, 
    /// Port provides outgoing signals
    Output, 
    /// Port can accept and provide signals
    Inout 
}
impl PortType {
    /// A function to check if the port type accepts input signals.
    pub fn accepts_input(self) -> bool {
        matches!(self, PortType::Input | PortType::Inout)
    }
    /// A function to check if the port type provides outgoing signals.
    pub fn accepts_output(self) -> bool {
        matches!(self, PortType::Output | PortType::Inout)
    }
}
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash, Debug)]
/// PortProperties defines the properties of a port for a digital logic component.
/// - `ty`: Declares what type of port it is (Input, Output, Inout)
/// - `bitsize`: Declares the size of the data the port works with in bits. 
pub struct PortProperties {
    /// Type of the port (Input, Output, Inout)
    pub ty: PortType,
    /// Size of the data the port works with in bits
    pub bitsize: u8
}

/// PortUpdate is a data structure that represents an update to a port's value during simulation.
/// - `index`: The index of the port that is being updated. I.e, in the port list, index of 0 is the first port.
/// - `value`: Represents the new value that will be assigned to the port at the given index.
pub struct PortUpdate {
    /// Index of the port being updated
    pub index: usize,
    /// New value to be assigned to the port
    pub value: BitArray
}

pub trait Component {
    fn ports(&self) -> Vec<PortProperties>;
    fn initialize(&self, _state: &mut [BitArray]) {}
    #[must_use]
    fn run(&self, old_inp: &[BitArray], inp: &[BitArray]) -> Vec<PortUpdate>;
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Sensitivity {
    Anyedge, Posedge, Negedge, DontCare
}
impl Sensitivity {
    pub fn activated(self, old: BitArray, new: BitArray) -> bool {
        assert_eq!(old.len(), new.len(), "Bit length should be the same");
        match self {
            Sensitivity::Anyedge  => old != new,
            Sensitivity::Posedge  => old.all(BitState::Low) && new.all(BitState::High),
            Sensitivity::Negedge  => old.all(BitState::High) && new.all(BitState::Low),
            Sensitivity::DontCare => false,
        }
    }
    pub fn any_activated(self, old: &[BitArray], new: &[BitArray]) -> bool {
        assert_eq!(old.len(), new.len(), "Array size should be the same");
        std::iter::zip(old, new)
            .any(|(&o, &n)| self.activated(o, n))
    }
}

fn port_list(config: &[(PortProperties, u8)]) -> Vec<PortProperties> {
    config.iter()
        .flat_map(|&(props, ct)| std::iter::repeat_n(props, usize::from(ct)))
        .collect()
}
macro_rules! decl_component_enum {
    ($ComponentEnum:ident: $($Component:ident),*$(,)?) => {
        pub enum $ComponentEnum {
            $($Component($Component)),*
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
            fn run(&self, old_inp: &[BitArray], inp: &[BitArray]) -> Vec<PortUpdate> {
                match self {
                    $(
                        Self::$Component(c) => c.run(old_inp, inp),
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
    And, Or, Xor, Nand, Nor, Xnor, Not, TriState, 
    Mux, Demux, Decoder, Splitter, Register
);

pub const MIN_GATE_INPUTS: u8 = 2;
pub const MAX_GATE_INPUTS: u8 = 64;
pub struct GateProperties {
    bitsize: u8,
    n_inputs: u8
}

macro_rules! gates {
    ($($Id:ident: $f:expr),*$(,)?) => {
        $(
            pub struct $Id {
                props: GateProperties
            }
            impl $Id {
                pub fn new(mut bitsize: u8, mut n_inputs: u8) -> Self {
                    bitsize = bitsize.clamp(BitArray::MIN_BITSIZE, BitArray::MAX_BITSIZE);
                    n_inputs = n_inputs.clamp(MIN_GATE_INPUTS, MAX_GATE_INPUTS);
                    Self { props: GateProperties { bitsize, n_inputs }}
                }
            }
            impl Component for $Id {
                fn ports(&self) -> Vec<PortProperties> {
                    port_list(&[
                        // inputs
                        (PortProperties { ty: PortType::Input, bitsize: self.props.bitsize }, self.props.n_inputs),
                        // outputs
                        (PortProperties { ty: PortType::Output, bitsize: self.props.bitsize }, 1),
                    ])
                }
                fn run(&self, _old_inp: &[BitArray], inp: &[BitArray]) -> Vec<PortUpdate> {
                    let value = inp[..usize::from(self.props.n_inputs)].iter()
                        .cloned()
                        .reduce($f)
                        .unwrap_or_else(|| BitArray::unknown(self.props.bitsize));
    
                    vec![PortUpdate { index: usize::from(self.props.n_inputs), value }]
                }
            }
        )*
    }
}

gates! {
    And:  |a, b| a & b,
    Or:   |a, b| a | b,
    Xor:  |a, b| a ^ b,
    Nand: |a, b| !(a & b),
    Nor:  |a, b| !(a | b),
    Xnor: |a, b| !(a ^ b),
}

pub struct BufNotProperties {
    bitsize: u8
}
pub struct Not {
    props: BufNotProperties
}
impl Not {
    pub fn new(mut bitsize: u8) -> Self {
        bitsize = bitsize.clamp(BitArray::MIN_BITSIZE, BitArray::MAX_BITSIZE);
        Self { props: BufNotProperties { bitsize }}
    }
}
impl Component for Not {
    fn ports(&self) -> Vec<PortProperties> {
        port_list(&[
            // input
            (PortProperties { ty: PortType::Input, bitsize: self.props.bitsize }, 1),
            // output
            (PortProperties { ty: PortType::Output, bitsize: self.props.bitsize }, 1),
        ])
    }

    fn run(&self, _old_inp: &[BitArray], inp: &[BitArray]) -> Vec<PortUpdate> {
        vec![PortUpdate { index: 1, value: !inp[0] }]
    }
}

pub struct TriState {
    props: BufNotProperties
}
impl TriState {
    pub fn new(mut bitsize: u8) -> Self {
        bitsize = bitsize.clamp(BitArray::MIN_BITSIZE, BitArray::MAX_BITSIZE);
        Self { props: BufNotProperties { bitsize }}
    }
}
impl Component for TriState {
    fn ports(&self) -> Vec<PortProperties> {
        port_list(&[
            // selector
            (PortProperties { ty: PortType::Input, bitsize: 1 }, 1),
            // input
            (PortProperties { ty: PortType::Input, bitsize: self.props.bitsize }, 1),
            // output
            (PortProperties { ty: PortType::Output, bitsize: self.props.bitsize }, 1),
        ])
    }

    fn run(&self, _old_inp: &[BitArray], inp: &[BitArray]) -> Vec<PortUpdate> {
        let gate = inp[0].index(0);
        let result = match gate {
            BitState::High => inp[1],
            BitState::Low | BitState::Imped => BitArray::floating(self.props.bitsize),
            BitState::Unk => BitArray::unknown(self.props.bitsize),
        };
        vec![PortUpdate { index: 2, value: result }]
    }
}

pub const MIN_SELSIZE: u8 = 1;
pub const MAX_SELSIZE: u8 = 8;
pub struct MuxProperties {
    bitsize: u8,
    selsize: u8
}
pub struct Mux {
    props: MuxProperties
}
impl Mux {
    pub fn new(mut bitsize: u8, mut selsize: u8) -> Self {
        bitsize = bitsize.clamp(BitArray::MIN_BITSIZE, BitArray::MAX_BITSIZE);
        selsize = selsize.clamp(MIN_SELSIZE, MAX_SELSIZE);
        Self { props: MuxProperties { bitsize, selsize }}
    }
}
impl Component for Mux {
    fn ports(&self) -> Vec<PortProperties> {
        port_list(&[
            // selector
            (PortProperties { ty: PortType::Input, bitsize: self.props.selsize }, 1),
            // inputs
            (PortProperties { ty: PortType::Input, bitsize: self.props.bitsize }, 1 << self.props.selsize),
            // output
            (PortProperties { ty: PortType::Output, bitsize: self.props.bitsize }, 1),
        ])
    }

    fn run(&self, _old_inp: &[BitArray], inp: &[BitArray]) -> Vec<PortUpdate> {
        let m_sel = u64::try_from(inp[0]);
        let result = match m_sel {
            Ok(sel) => inp[sel as usize + 1],
            Err(e) => BitArray::repeat(e.bit_state(), self.props.bitsize),
        };
        vec![PortUpdate { index: (1 << self.props.selsize) + 1, value: result }]
    }
}
pub struct Demux {
    props: MuxProperties
}
impl Demux {
    pub fn new(mut bitsize: u8, mut selsize: u8) -> Self {
        bitsize = bitsize.clamp(BitArray::MIN_BITSIZE, BitArray::MAX_BITSIZE);
        selsize = selsize.clamp(MIN_SELSIZE, MAX_SELSIZE);
        Self { props: MuxProperties { bitsize, selsize }}
    }
}
impl Component for Demux {
    fn ports(&self) -> Vec<PortProperties> {
            port_list(&[
            // selector
            (PortProperties { ty: PortType::Input, bitsize: self.props.selsize }, 1),
            // input
            (PortProperties { ty: PortType::Input, bitsize: self.props.bitsize }, 1),
            // outputs
            (PortProperties { ty: PortType::Output, bitsize: self.props.bitsize }, 1 << self.props.selsize),
        ])
    }
    fn run(&self, _old_inp: &[BitArray], inp: &[BitArray]) -> Vec<PortUpdate> {
        let m_sel = u64::try_from(inp[0]);
        let result = match m_sel {
            Ok(sel) => {
                let mut result = vec![BitArray::repeat(BitState::Low, self.props.bitsize); 1 << self.props.selsize];
                result[sel as usize] = inp[1];
                result
            },
            Err(e) => vec![BitArray::repeat(e.bit_state(), self.props.bitsize); 1 << self.props.selsize],
        };

        result.into_iter()
            .enumerate()
            .map(|(i, value)| PortUpdate { index: 2 + i, value })
            .collect()
    }
}

pub struct DecoderProperties {
    selsize: u8
}
pub struct Decoder {
    props: DecoderProperties
}
impl Decoder {
    pub fn new(mut selsize: u8) -> Self {
        selsize = selsize.clamp(MIN_SELSIZE, MAX_SELSIZE);
        Self { props: DecoderProperties { selsize }}
    }
}
impl Component for Decoder {
    fn ports(&self) -> Vec<PortProperties> {
        port_list(&[
            // selector
            (PortProperties { ty: PortType::Input, bitsize: self.props.selsize }, 1),
            // outputs
            (PortProperties { ty: PortType::Output, bitsize: 1 }, 1 << self.props.selsize),
        ])
    }

    fn run(&self, _old_inp: &[BitArray], inp: &[BitArray]) -> Vec<PortUpdate> {
        let m_sel = u64::try_from(inp[0]);
        let result = match m_sel {
            Ok(sel) => {
                let mut result = vec![BitArray::from_iter([BitState::Low]); 1 << self.props.selsize];
                result[sel as usize] = BitArray::from_iter([BitState::High]);
                result
            },
            Err(e) => vec![BitArray::from_iter([e.bit_state()]); 1 << self.props.selsize],
        };

        result.into_iter()
            .enumerate()
            .map(|(i, value)| PortUpdate { index: 1 + i, value })
            .collect()
    }
}

pub struct Splitter {
    props: BufNotProperties
}
impl Splitter {
    pub fn new(mut bitsize: u8) -> Self {
        bitsize = bitsize.clamp(BitArray::MIN_BITSIZE, BitArray::MAX_BITSIZE);
        Self { props: BufNotProperties { bitsize } }
    }
}
impl Component for Splitter {
    fn ports(&self) -> Vec<PortProperties> {
        port_list(&[
            // joined
            (PortProperties { ty: PortType::Inout, bitsize: self.props.bitsize }, 1),
            // split
            (PortProperties { ty: PortType::Inout, bitsize: 1 }, self.props.bitsize),
        ])
    }

    fn run(&self, old_inp: &[BitArray], inp: &[BitArray]) -> Vec<PortUpdate> {
        if Sensitivity::Anyedge.activated(old_inp[0], inp[0]) {
            std::iter::zip(1.., inp[0])
                .map(|(index, bit)| PortUpdate { index, value: BitArray::from_iter([bit]) })
                .collect()
        } else if Sensitivity::Anyedge.any_activated(&old_inp[1..], &inp[1..]) {
            let value = inp[1..].iter()
                .map(|b| b.index(0))
                .collect();
            vec![PortUpdate { index: 0, value }]
        } else {
            vec![]
        }
    }
}

pub struct Register {
    props: BufNotProperties
}
impl Register {
    pub fn new(mut bitsize: u8) -> Self {
        bitsize = bitsize.clamp(BitArray::MIN_BITSIZE, BitArray::MAX_BITSIZE);
        Self { props: BufNotProperties { bitsize } }
    }
}
impl Component for Register {
    fn ports(&self) -> Vec<PortProperties> {
        port_list(&[
            // din
            (PortProperties { ty: PortType::Input, bitsize: self.props.bitsize }, 1),
            // enable, clock, clear
            (PortProperties { ty: PortType::Input, bitsize: 1 }, 3),
            // dout
            (PortProperties { ty: PortType::Output, bitsize: self.props.bitsize }, 1),
        ])
    }
    fn initialize(&self, state: &mut [BitArray]) {
        state[4] = BitArray::repeat(BitState::Low, self.props.bitsize);
    }
    fn run(&self, old_inp: &[BitArray], inp: &[BitArray]) -> Vec<PortUpdate> {
        if inp[3].all(BitState::High) {
            vec![PortUpdate { index: 4, value: BitArray::repeat(BitState::Low, self.props.bitsize) }]
        } else if Sensitivity::Posedge.activated(old_inp[2], inp[2]) && inp[1].all(BitState::High) {
            vec![PortUpdate { index: 4, value: inp[0] }]
        } else {
            vec![]
        }
    }
}