# Breath-hold validation protocol

Use for any "does the apnoea pipeline work" check, on self or paediatric phantom.
Safe in a chair, no ethics needed.

1. Recording starts. 30 s of normal breathing.
2. Operator/subject manually logs `rr=0` to mark hold start.
   Hold breath for 30–60 s.
3. Operator/subject manually logs `rr=15` (or whatever resumes)
   to mark recovery start. Resume normal breathing for 30+ s.
4. Recording ends.

Single recording yields steady-state + cessation + recovery data points
(3–4 paired comparisons against ground truth). Save raw frames + manual
phase timestamps in the run dir for replay analysis.
