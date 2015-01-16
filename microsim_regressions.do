***************************
** Micro-simualation Analysis of an agent-based percolation model of innovation
** Nathan Goldschlag
** January 15, 2015
** Version 1.0

** This .do file executes the regressions and difference in means tests for micro-simulation output.
***************************

global workDir "C:\Users\ngold\documents\python_library\work_percolation"
insheet using "$workDir\2014-10-26 2039_34_microSim_percol_params.csv", comma clear
*rename run run_1
tempfile params
save "`params'"

insheet using "$workDir\2014-10-26 2039_34_microSim_percol_rundata.csv", comma clear
*rename run run_1
merge 1:1 run using "`params'"
drop _merge

replace totalpatents = "." if totalpatents=="N/A"
destring totalpatents, replace
replace perbpfpat = "." if perbpfpat=="N/A"
destring perbpfpat, replace
replace dopatent = "1" if dopatent=="True"
replace dopatent = "0" if dopatent=="False"
destring dopatent, replace
gen dopatdecay = dopatent*pidecayrate
replace patentlife = . if dopatent == 0
replace patentradius = . if dopatent == 0
gen lmaxbpf = log(maxbpf)

** regressions
reg maxbpf dopatent resistmu pidecayrate clusterindex, vce(robust)
outreg2 using $workDir\percolregs.doc, replace ctitle(maxBPF)
reg meanbpf dopatent resistmu pidecayrate clusterindex, vce(robust)
outreg2 using $workDir\percolregs.doc, append ctitle(meanBPF)
reg numinnov dopatent resistmu pidecayrate clusterindex, vce(robust)
outreg2 using $workDir\percolregs.doc, append ctitle(numInnov)

reg maxbpf resistmu pidecayrate patentlife patentradius clusterindex if dopatent==1, vce(robust)
outreg2 using $workDir\percolregs.doc, append ctitle(maxBPF With Patents)
reg meanbpf resistmu pidecayrate patentlife patentradius clusterindex if dopatent==1, vce(robust)
outreg2 using $workDir\percolregs.doc, append ctitle(meanBPF With Patents)
reg numinnov resistmu pidecayrate patentlife patentradius clusterindex if dopatent==1, vce(robust)
outreg2 using $workDir\percolregs.doc, append ctitle(numInnov With Patents)


** diff in means of logs, lognormal
ttest lmaxbpf, by(dopatent)
ttest lmaxbpf, by(dopatent) unequal 
* simple tech space
ttest lmaxbpf if resistmu<2.25, by(dopatent) unequal
* complex tech space
ttest lmaxbpf if resistmu>3.5, by(dopatent) unequal
* high monopoly power
ttest lmaxbpf if pidecayrate<0.04, by(dopatent) unequal
* low monopoly power
ttest lmaxbpf if pidecayrate>0.07, by(dopatent) unequal
