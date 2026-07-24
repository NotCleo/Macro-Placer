# Macro-Placer
Macro-Placer is our custom implementation of placing macro cells on a floorplan for Macro


This repository contains our submission to the macro place challenge 2026 hosted by Partcl x Hudson River Trading

    See more : https://github.com/partcleda/macro-place-challenge-2026

Our Complete implementation method :

    See more : https://medium.com/@amrutayan6/macro-placement-a73c399e64fa

Overview of the challenge : 

Macro placement is the problem of positioning large fixed-size blocks (SRAMs, IPs, analog macros, etc.) on a chip floorplan so that routing congestion, timing, power delivery, and area constraints are balanced. Unlike standard-cell placement, macros have strong geometric and connectivity constraints, so the challenge is to explore a highly discrete design space while minimizing wirelength, avoiding blockages, and preserving downstream routability and timing quality.
  
    Our team (Two-IIITK-Kids) currently is on global rank 13 :D
    
The ranking was produced after taking in three metrics {congestion, wire-length, density} to produce a proxy cost;
    
    Proxy Cost = 1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion
    
Our team produced the following performance for the ICCAD04 benchmark;

    (Official) Proxy Cost = 1.0295, Time = 2432s/benchmark
    (Unofficial) Proxy Cost = 0.9767, Time = 3160s/benchmark


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


# 1. Clone the challenge repository
    git clone https://github.com/partcleda/partcl-macro-place-challenge.git
    cd MacroPlacer

# 2. Initialize TILOS MacroPlacement submodule
    git submodule update --init external/MacroPlacement

# 3. Install python dependencies via uv
    uv sync
    uv pip install numba scipy

    sudo apt-get update && sudo apt-get install -y libfftw3-dev pkg-config

# 4. Run the evaluation for ibm01 benchmark
    uv run evaluate Two-IIITK-Kids/placer.py -b ibm01

# 5. Run the evaluation for ibm01 benchmark
    uv run evaluate Two-IIITK-Kids/placer.py --all

    

