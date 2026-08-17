# Macro-Placer

Macro-Placer is our custom implementation of placing macro cells on an ASIC floorplan


### This repository contains our submission to the macro place challenge 2026 hosted by Partcl x Hudson River Trading

    See more : https://github.com/partcleda/macro-place-challenge-2026

## Our Complete implementation method :

    See more : https://medium.com/@amrutayan6/macro-placement-a73c399e64fa

## Overview of the challenge : 

Macro placement is the problem of positioning large fixed-size blocks (SRAMs, IPs, analog macros, etc.) on a chip floorplan so that routing congestion, timing, power delivery, and area constraints are balanced. Unlike standard-cell placement, macros have strong geometric and connectivity constraints, so the challenge is to explore a highly discrete design space while minimizing wirelength, avoiding blockages, and preserving downstream routability and timing quality.
  
    Our team (Two-IIITK-Kids) currently is on global rank 13 :D
    
#### The ranking was produced after taking in three metrics {congestion, wire-length, density} to produce a proxy cost;
    
    Proxy Cost = 1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion
    
#### Our team produced the following performance for the ICCAD04 benchmark;

    (Official) Proxy Cost = 1.0295, Time = 2432s/benchmark
    (Unofficial) Proxy Cost = 0.9767, Time = 3160s/benchmark


#### Result summary (ICCAD 04)
        
        --------------------------------------------------------------------------------
            Benchmark     Proxy        SA   RePlAce     vs SA  vs RePlAce  Overlaps
        --------------------------------------------------------------------------------
                ibm01    0.7496    1.3166    0.9976     43.1%       24.9%         0
                ibm02    0.9463    1.9072    1.8370     50.4%       48.5%         0
                ibm03    0.8727    1.7401    1.3222     49.8%       34.0%         0
                ibm04    0.8919    1.5037    1.3024     40.7%       31.5%         0
                ibm06    1.1173    2.5057    1.6187     55.4%       31.0%         0
                ibm07    0.9627    2.0229    1.4633     52.4%       34.2%         0
                ibm08    1.0078    1.9239    1.4285     47.6%       29.5%         0
                ibm09    0.7712    1.3875    1.1194     44.4%       31.1%         0
                ibm10    1.0680    2.1108    1.5009     49.4%       28.8%         0
                ibm11    0.7570    1.7111    1.1774     55.8%       35.7%         0
                ibm12    1.1126    2.8261    1.7261     60.6%       35.5%         0
                ibm13    0.8466    1.9141    1.3355     55.8%       36.6%         0
                ibm14    1.0519    2.2750    1.5436     53.8%       31.9%         0
                ibm15    1.0978    2.3000    1.5159     52.3%       27.6%         0
                ibm16    1.0041    2.2337    1.4780     55.0%       32.1%         0
                ibm17    1.1894    3.6726    1.6446     67.6%       27.7%         0
                ibm18    1.1567    2.7755    1.7722     58.3%       34.7%         0
        --------------------------------------------------------------------------------
                  AVG    0.9767    2.1251    1.4578     54.0%       33.0%         0
        
        Total runtime: 55572.60s


  The Repository Structure : 

    .
    ├── Dockerfile                  # Runtime environment
    ├── requirements.txt            # Python Packages
    ├── uv.lock                     # Python Package manager
    ├── SETUP.md                    # Environment Setup
    ├── SCORING.md                  # Challenge scoring rules
    ├── Challenge-README.md         # Challenge Description
    ├── README.md                   # This file
    ├── LICENSE                     # Apache 2.0
    ├── benchmarks                  # benchmark files
    ├── external                    # benchmark files
    ├── baselines                   # baseline files
    ├── eval_docker                 # Docker environment
    ├── macro_place                 # Support scripts
    ├── scripts                     # Support scripts
    ├── test                        # Support scripts
    ├── src                         # Support scripts
    ├── Two-IIITK-Kids              # [Our Submission]
    │   ├── placer.py               # Entry point script
    │   ├── script1.py            
    │   ├── script2.py            
    │   ├── script3.py            
    │   ├── script4.py            
    │   ├── script5.py           
    │   ├── script6.py            
    │   ├── script7.py            
    │   ├── script8.py           
    │   ├── script9.py          
    │   ├── script10.py           
    │   ├── script11.py           
    │   ├── script12.py           
    │   ├── script13.py           
    │   ├── script14.py          
    │   ├── script15.py          
    │   ├── script16.py          
    │   ├── script17.py           
    │   ├── script18.py          
    │   ├── script19.py          
    │   ├── script20.py          
    │   ├── script21.py          
    └──  └── Performance/performance.txt

I personally recommend the non-Docker method of testing; 

##  Evaluation Instructions (Manual — Without Docker)


#### 1. Clone the challenge repository
    git clone https://github.com/NotCleo/macro-placer.git
    cd macro-placer

#### 2. Initialize TILOS MacroPlacement submodule
    git submodule update --init external/MacroPlacement

#### 3. Install python dependencies via uv
    uv sync
    uv pip install numba scipy

    sudo apt-get update && sudo apt-get install -y libfftw3-dev pkg-config

#### 4. Run the evaluation for ibm01 benchmark
    uv run evaluate Two-IIITK-Kids/placer.py -b ibm01

#### 5. Run the evaluation for ibm01 benchmark
    uv run evaluate Two-IIITK-Kids/placer.py --all

    

