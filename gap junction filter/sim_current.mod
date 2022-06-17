INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}
NEURON {
     POINT_PROCESS sim_current
     ELECTRODE_CURRENT I
	 RANGE tau1,tau2,g,gmax,I,intensity,stime
	 
	 }
UNITS {
     (pA) = (picoamp)
	 (nA) = (nanoamp)
     (mV) = (millivolt)
     (pS) = (picosiemens)
     (umho) = (micromho)
     (uM) = (micro/liter)
	 FARADAY = (faraday) (coulomb)
}

PARAMETER {
g=0.04
gmax
rh
intensity
tau1=64(ms)
tau2=68(ms)
stime=100 (ms)
}
ASSIGNED{
I (nA)
}

BREAKPOINT {
if(t>=stime){
	:rh=0.0625*intensity
	:gmax=0.04/(1+exp(-(rh-8)))
	gmax=0.00015686*intensity
	g=0.04-gmax*(exp(-(t-stime)/tau1)-exp(-(t-stime)/tau2))/(((tau1 - tau2)*(tau2/tau1)^(tau1/(tau1 - tau2)))/tau2)
	I=g
	}
else{
	I=g
	}

}

