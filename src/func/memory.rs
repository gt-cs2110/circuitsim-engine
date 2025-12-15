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

/// A ROM component

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
                // Address in
                (PortProperties {ty: PortType::Input, bitsize: self.addrsize}, 1), 
                // Enable
                (PortProperties {ty: PortType::Input, bitsize: 1}, 1), 
                // Data out
                (PortProperties {ty: PortType::Output, bitsize: self.bitsize}, 1) 
            ]
        )
    }

    fn initialize_inner_state(&self,_graphs: &CircuitGraphMap) -> Option<crate::circuit::state::InnerFunctionState> {
        let size = 1 << self.addrsize;
        Some(InnerFunctionState::Rom(vec![0u64; size]))
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

         let Some(InnerFunctionState::Rom(mem)) = ctx.inner_state else {
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
                // TODO: maybe change this into float/unk or a real assertion
            },
            Err(e) => {
                BitArray::repeat(e.bit_state(), self.bitsize) // just spit the bits back out the other end
            }

        };

        vec![PortUpdate { index: 2, value: output }]
    }

}