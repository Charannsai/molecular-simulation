"""One shared unit registry, so every quantity in the simulator carries real units.

pint catches unit mistakes (adding joules to watts, mixing GB and GiB) at the moment they happen,
which is the most common way a "1,000x better" result sneaks into a paper.
"""

import pint

ureg = pint.UnitRegistry()
ureg.formatter.default_format = "~P"  # short, pretty units: "GB/s", "mJ"

# Counting units for LLM work, so "per token" and "per operation" are checked like any other unit.
ureg.define("token = [token] = tok")
ureg.define("operation = [operation] = op")
ureg.define("MAC = 2 * operation")  # one multiply-accumulate = two operations

Q = ureg.Quantity
