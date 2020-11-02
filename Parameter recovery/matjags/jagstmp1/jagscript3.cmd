load wiener
load glm
load dic
model in "C:\Users\benja\OneDrive\Dokumenter\GitHub\Master-thesis\Parameter recovery\JAGS\JAGS_models_Subjectwise_simulation.txt"
data in jagsdata.R
compile, nchains(1)
parameters in jagsinit3.R
initialize
update 100
monitor set dx1, thin(1)
monitor set dx2, thin(1)
monitor set dx3, thin(1)
monitor set dx4, thin(1)
monitor set p_a1, thin(1)
monitor set p_a2, thin(1)
monitor set p_b1, thin(1)
monitor set p_b2, thin(1)
monitor set y_pt, thin(1)
monitor set alpha_pt, thin(1)
monitor set gamma_pt, thin(1)
monitor set delta_pt, thin(1)
monitor set beta_pt, thin(1)
monitor set y_lml, thin(1)
monitor set alpha_lml, thin(1)
monitor set gamma_lml, thin(1)
monitor set delta_lml, thin(1)
monitor set beta_lml, thin(1)
monitor deviance
update 50
coda *, stem('CODA3')