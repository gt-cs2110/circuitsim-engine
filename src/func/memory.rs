use std::vec;

use crate::bitarr;
use crate::bitarray::{BitArray, BitState};
use crate::circuit::CircuitGraphMap;
use crate::circuit::state::InnerFunctionState;
use crate::func::{Component, PortProperties, PortType, PortUpdate, RunContext, Sensitivity, port_list};

/// A register component.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct Register {
    bitsize: u8
}
impl Register {
    /// Creates a new instance of the Register with specified bitsize.
    pub fn new(bitsize: u8) -> Self {
        Self {
            bitsize: bitsize.clamp(BitArray::MIN_BITSIZE, BitArray::MAX_BITSIZE)
        }
    }
}
impl Component for Register {
    fn ports(&self, _: &CircuitGraphMap) -> Vec<PortProperties> {
        port_list(&[
            // din
            (PortProperties { ty: PortType::Input, bitsize: self.bitsize }, 1),
            // enable, clock, clear
            (PortProperties { ty: PortType::Input, bitsize: 1 }, 3),
            // dout
            (PortProperties { ty: PortType::Output, bitsize: self.bitsize }, 1),
        ])
    }
    fn initialize_port_state(&self, state: &mut [BitArray]) {
        state[4] = bitarr![0; self.bitsize];
    }
    fn run_inner(&self, ctx: RunContext<'_>) -> Vec<PortUpdate> {
        if ctx.new_ports[3].all(BitState::High) {
            vec![PortUpdate { index: 4, value: bitarr![0; self.bitsize] }]
        } else if Sensitivity::Posedge.activated(ctx.old_ports[2], ctx.new_ports[2]) && ctx.new_ports[1].all(BitState::High) {
            vec![PortUpdate { index: 4, value: ctx.new_ports[0] }]
        } else {
            vec![]
        }
    }
}

/// A ROM component.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct Rom {
    bitsize: u8,
    addrsize: u8 
}

impl Rom {
    /// Creates a new instance of ROM 
    /// 
    /// The initial contents, `mem` will be resized to 2^addrsize padded with 0s if necessary
    pub fn new(bitsize: u8, addrsize: u8, mem: Vec<u64>) -> Self {
        let addrsize = addrsize.clamp(1, 20); // This is CircuitSim's current limit, but we could change it
        let bitsize = bitsize.clamp(BitArray::MIN_BITSIZE,BitArray::MAX_BITSIZE);

        let mut memory = mem;
        memory.resize(1 << addrsize, 0); // clamp mem size/contents

        Self {
            bitsize,
            addrsize
        }
    }
}

impl Component for Rom {
    fn ports(&self, _graphs: &CircuitGraphMap) -> Vec<PortProperties> {
        port_list(
            &[
                // address
                (PortProperties {ty: PortType::Input, bitsize: self.addrsize}, 1), 
                // enable
                (PortProperties {ty: PortType::Input, bitsize: 1}, 1), 
                // dout
                (PortProperties {ty: PortType::Output, bitsize: self.bitsize}, 1) 
            ]
        )
    }

    fn initialize_inner_state(&self,_graphs: &CircuitGraphMap) -> Option<crate::circuit::state::InnerFunctionState> {
        let size = 1 << self.addrsize;
        Some(InnerFunctionState::Memory(vec![0u64; size]))
    }

    fn run_inner(&self,ctx:RunContext<'_>) -> Vec<PortUpdate> {
        // Check whether ROM is enabled first
        let enable = ctx.new_ports[1].index(0); 
        if enable == BitState::High {
            return vec![PortUpdate {
                index: 2,
                value: bitarr![Z; self.bitsize]
            }];
        }

         let Some(InnerFunctionState::Memory(mem)) = ctx.inner_state else {
            unreachable!("ROM's inner state was not a Vec<u64>");
            // TODO: I asked chat about what to put here ^ but idk if this is right i think its right
        };

        let try_addr = u64::try_from(ctx.new_ports[0]);

        let output = match try_addr {
            Ok(addr) if addr < (mem.len() as u64) => { // Assert that address is within the bounds of this ROM
                BitArray::from_bits(mem[addr as usize], self.bitsize)
            },
            Ok(_addr) => {
                // not floating but not in bounds either
                bitarr![0; self.bitsize]
                // Return 0s 
                // TODO: maybe change this into float/unk or an assertion
            },
            Err(e) => {
                BitArray::repeat(e.bit_state(), self.bitsize) // just spit the bits back out the other end
            }

        };

        vec![PortUpdate { index: 2, value: output }]
    }

}

/// A RAM component.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct Ram {
    bitsize: u8,
    addrsize: u8
}

impl Ram {
    /// Creates a new instance of RAM 
    /// 
    /// The initial contents, `mem` will be resized to 2^addrsize padded with 0s if necessary
    pub fn new(bitsize: u8, addrsize: u8, mem: Vec<u64>) -> Self {
        let addrsize = addrsize.clamp(1, 20); // This is CircuitSim's current limit, but we could change it
        let bitsize = bitsize.clamp(BitArray::MIN_BITSIZE,BitArray::MAX_BITSIZE);

        let mut memory = mem;
        memory.resize(1 << addrsize, 0); // clamp mem size/contents

        Self {
            bitsize,
            addrsize
        }
    }
}

impl Component for Ram {
    fn ports(&self, _: &CircuitGraphMap) -> Vec<PortProperties> {
        port_list(&[
            // address
            (PortProperties { ty: PortType::Input, bitsize: self.addrsize }, 1),
            // data in/out
            (PortProperties { ty: PortType::Inout, bitsize: self.bitsize }, 1),
            // clock
            (PortProperties { ty: PortType::Input, bitsize: 1 }, 1),
            // enable
            (PortProperties { ty: PortType::Input, bitsize: 1 }, 1),
            // load/!store
            (PortProperties { ty: PortType::Input, bitsize: 1 }, 1),
            // clear
            (PortProperties { ty: PortType::Input, bitsize: 1 }, 1),
        ])
    }

    fn initialize_inner_state(&self,_graphs: &CircuitGraphMap) -> Option<crate::circuit::state::InnerFunctionState> {
        let size = 1 << self.addrsize;
        Some(InnerFunctionState::Memory(vec![0u64; size]))
    }
    fn run_inner(&self, ctx: RunContext<'_>) -> Vec<PortUpdate> {
        let address = ctx.new_ports[0];
        let data = ctx.new_ports[1];
        let clock = ctx.new_ports[2];
        let enable = ctx.new_ports[3];
        let load = ctx.new_ports[4];
        let clear = ctx.new_ports[5];

        let Some(InnerFunctionState::Memory(mem)) = ctx.inner_state else {
            unreachable!("RAM's inner state was not a Vec<u64>");
            // TODO: I asked chat about what to put here ^ but idk if this is right i think its right
        };
        
        // Check clear input
        if clear.all(BitState::High) {
            mem.fill(0);
            return vec![PortUpdate { 
                index: 1, 
                value: bitarr![Z; self.bitsize] 
            }];
        }

        // Check enable 
        if enable.all(BitState::Low) {
            return vec![PortUpdate { 
                index: 1, 
                value: bitarr![Z; self.bitsize] 
            }];
        }

        // Check load signal
        let load_state = load.index(0);
        match load_state {
            BitState::High => {
                // Load mode 
                let addr_result = u64::try_from(address);
                let output = match addr_result {
                    Ok(addr) if (addr as usize) < mem.len() => {
                        BitArray::from_bits(mem[addr as usize], self.bitsize)
                    },
                    Ok(_addr) => {
                        // not floating but not in bounds either
                        bitarr![0; self.bitsize]
                        // Return 0s 
                        // TODO: maybe change this into float/unk or an assertion
                    },
                    Err(e) => {
                        BitArray::repeat(e.bit_state(), self.bitsize)
                    }
                };
                vec![PortUpdate { index: 1, value: output }]
            },
            BitState::Low => {
                // Store mode 
                if Sensitivity::Posedge.activated(ctx.old_ports[2], clock) {
                    let addr_result = u64::try_from(address);
                    let data_result = u64::try_from(data);
                    
                    if let (Ok(addr), Ok(data)) = (addr_result, data_result) && 
                        (addr as usize) < mem.len() {

                        mem[addr as usize] = data;
                    }
                    // After write, output Z (not driving)
                }
                vec![PortUpdate { 
                    index: 1, 
                    value: bitarr![Z; self.bitsize] 
                }]
            },
            BitState::Unk | BitState::Imped => {
                vec![PortUpdate { 
                    index: 1, 
                    value: bitarr![X; self.bitsize] 
                }]
            }
        }
    }
}