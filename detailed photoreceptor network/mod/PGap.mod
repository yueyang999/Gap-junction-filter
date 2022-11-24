TITLE Gap junction current for parallel computation

COMMENT
Implemented by Erin Munro, 
copied largely from:
The NEURON Book by Nicholas T. Carnevale, Michael L. Hines
online at Google books
ENDCOMMENT

UNITS { 
	(mV) = (millivolt) 
	(nA) = (nanoamp) 
  (uS) = (microsiemens)
} 

NEURON { 
    POINT_PROCESS PGap
    RANGE vgap
	  RANGE g, i
	  NONSPECIFIC_CURRENT i
}

PARAMETER { 
	  g = 0.0 	   (uS)
} 

NET_RECEIVE(weight (uS)){
    
}

ASSIGNED {
    v (mV)
    vgap (mV)
    i (nA)
}

BREAKPOINT { 
	  i = g*(v - vgap) 
} 
