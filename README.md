# Stochastic Causal Net Miner

Stochastic Causal Net Miner discovers of stochastic causal nets. The inputs are an event log and a causal net model (file format should be the same as exported file ProM), and the output is a stochastic causal net. The two current implemented plugins adopt the techniques introduced in the following to assist binding weights estimation. 

* Stochastic Markovian Abstraction:
Rocha, E.G., Leemans, S.J.J., van der Aalst, W.M.P.: Stochastic conformance checking based on expected subtrace frequency. In: ICPM. pp. 73–80. IEEE (2024

* Unit Earth Mover Stochastic Conformance:
Leemans, S.J.J., Syring, A.F., van der Aalst, W.M.P.: Earth movers’ stochastic conformance checking. In: BPM Forum. Lecture Notes in Business Information Processing, vol. 360, pp. 127–143. Springer (2019)

## Usage
The following is the code snippet to use the Markovian abstraction-based stochastic discovery algorithm in optimization/scn_miner.py and optimization/approximated_scn_miner.py. 

Change the input and output path for your log and causal net to mine a stochastic causal net.

```python
if __name__ == "__main__":
    # Define file paths
    log_path = '../data/application.xes'
    model_path = '../data/application_hm.cnet'

    # Load the data
    log = xes_importer.apply(log_path)
    slang = get_stochastic_language(log)
    symbolic_cn = import_symbolic_causal_net_from_xml(model_path)
    
    # set target trace length for markovian abstraction
    k = 2
    
    # Perform stochastic discovery using Markovian abstraction
    optimize_with_k_th_uemsc(slang, symbolic_cn, k)
```
