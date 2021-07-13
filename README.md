# LSTMset
Implementation of parallel computation of a set of LSTMs.

Task: Run through N independent LSTMs in parallel.

Layer: LSTMCellset
Inputs: x, (hidden_state, cell_state) 
x: (batch, # of LSTMs, input_size)
hidden_state: (batch, # of LSTMs, hidden_size)
cell_state: (batch, # of LSTMs, hidden_size)
Outputs:
hidden_state: (batch, # of LSTMs, hidden_size)
cell_state: (batch, # of LSTMs, hidden_size)

Layer: LSTMsetLayer
Inputs: inputs, (hidden_state, cell_state) 
inputs: (batch, time, # of LSTMs, input_size)
hidden_state: (batch, # of LSTMs, hidden_size)
cell_state: (batch, # of LSTMs, hidden_size)
Outputs: (batch, time, # of LSTMs, hidden_size)
