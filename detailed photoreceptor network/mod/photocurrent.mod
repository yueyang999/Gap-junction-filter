INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}
NEURON {
     POINT_PROCESS photocurrent
     ELECTRODE_CURRENT Iphoto
	 RANGE Rh,B,Tr,PDE,Ca,Cab,cGMP,Iphoto,jhv,input,env,stime,htime,temp1,J
	 
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
STATE {
    Rh   (uM)                                   : Active rhodopsin. 
	B    (uM)                                   :  bleaching rhodopsin 
	Tr   (uM)                                   : active transducin (uM)
	PDE  (uM)                                   : active phosphodiesterase (uM)
	Ca   (uM)                                   : intracellular free Ca in outer segment (uM)
	Cab  (uM)                                   : intracellular Ca buffer in outer segment (uM)
	cGMP (uM)                                   : cyclic GMP (uM)
  
}

PARAMETER {
	tau_r=34 (ms)                             	: lifetime of R * (s) depends on the concentration of Ca ++ [Adjusted for rods with alpha2 values from Torre]
	tau_b0=25000  (ms)        		                : bleaching time constant recovery at very low brightness values (the entire range is dynamic and values between 25s - 150s) (Kenkre, Moran, Lamb y Mahroo en 2005).
	eps = 0.0005(/uM/ms)
	Ttot = 1000 (uM)
	b1 = 0.0025 (/ms)
	tau1 = 0.0002 (/uM/ms)
	tau2 = 0.005  (/ms)
	PDEtot = 100 (uM)
	rCa = 0.05 (/ms)
	c0 = 0.1 (uM)
	
	a1 = 0.05  (/ms)
	a2 = 0.0000003 (/ms)
	a3 = 0.00003 (/ms)
	b = 0.00025 (uM/ms/pA)
	k1 = 0.0002 (/uM/ms) : curve shape
	k2 = 0.0008  (/ms)    : curve shape
	eT = 500 (uM)
	Vbar = 0.0004(/ms) :balance point 0.00003(/ms)
	Kc = 0.1  (uM)
	K=10(uM)
	Amax= 0.0656 (uM/ms) :balance point 
	sigma = 0.001(/uM/ms) :amplitude
	Jmax = 5040 (pA)
	jhv (uM/ms)
	J (pA)
	temp1 (uM/ms)
	input=0 (uM/ms)
	env = 0 (uM/ms)
	stime=0 (ms)
	htime=10 (ms)
	}
ASSIGNED{
Iphoto (nA)
}
INITIAL {
	Rh = 0.0                   : Rh(0) Active rhodopsin.
	B = 0.0                    : B(0)  Bleaching rhodopsin.
	Tr = 0.0                   : Tr(0)
	PDE = 0.0                  : PDE(0)
	Ca = 0.3              : Ca(0)
	Cab = 34.9              : Cab(0)
	cGMP = 2             : cGMP(0)
}

BREAKPOINT {
	:if(t>=5000*k&&t<(5000*k+1)){
	:    jhv=0.0005*sin(t/(10*(k+1)))+0.0005
	:}
	if(t>=stime&&t<=htime){
		jhv=input+env
		}
	else{
		jhv=env
	}
	
	J = Jmax*cGMP^3 / (cGMP^3+K^3)
	SOLVE states METHOD derivimplicit
	Iphoto = (0.001)*J

}
DERIVATIVE states { 
	Rh' =jhv-a1*Rh+a2*B
	B' = a1*Rh-(a2+a3)*B
	Tr'  = eps*Rh*(Ttot-Tr) - b1*Tr -tau1*Tr*(PDEtot-PDE) + tau2*PDE
	PDE' = tau1*Tr*(PDEtot-PDE) - tau2*PDE
	Ca'  = b*J - rCa*(Ca-c0) - k1*(eT-Cab)*Ca + k2*Cab
	Cab' = k1*(eT-Cab)*Ca - k2*Cab
	temp1= Amax/(1.0+Ca^4/Kc^4)
	cGMP'= temp1- cGMP*(Vbar+sigma*PDE)
	}

  
