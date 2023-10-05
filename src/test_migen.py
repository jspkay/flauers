from migen import *
from migen.fhdl import verilog
from saffira.sahdl import PE

dut = PE(8, 32)

v = verilog.convert(
    dut,
    ios = {dut.a_in, dut.a_out, dut.b_out, dut.c_out, dut.b_in, dut.c_in}
)
print(v)
