SpinSystem RPC
{
	Spin electron1
	{
		spin = 1/2;
		type = electron;
		tensor = isotropic(2);
	}

	Spin electron2
	{
		spin = 1/2;
		type = electron;
		tensor = isotropic(2);
	}

	Spin nucleus1
	{
		spin = 1/2;
		type = nucleus;
		tensor = isotropic("1.0");
	}
	
	Interaction Hyperfine1
 	{
		type = Hyperfine;
 		group1 = electron1;
 		group2 = nucleus1;
		tensor= matrix(" 2.13814981 3.19255832 -2.48895215;
			3.19255832 15.45032887 -12.44778343;
			-2.48895215 -12.44778343 12.49532827");
		prefactor =3.568245455e-5;
 	}

	Interaction Zeeman
	{
		type = Zeeman;
		field = "5e-5 0 0";
		spins = electron1, electron2;
	}

	State SingletUP
	{
		spins(electron1, electron2) = |1/2,-1/2> - |-1/2,1/2>;
		spin(nucleus1) = |1/2>;
	}

	State SingletDOWN
	{
		spins(electron1, electron2) = |1/2,-1/2> - |-1/2,1/2>;
		spin(nucleus1) = |-1/2>;
	}

	State T0UP
	{
		spins(electron1, electron2) = |1/2,-1/2> + |-1/2,1/2>;
		spin(nucleus1) = |1/2>;
	}
	State T0DOWN
	{
		spins(electron1, electron2) = |1/2,-1/2> + |-1/2,1/2>;
		spin(nucleus1) = |-1/2>;
	}

	State TPlusUP
	{
		spins(electron1) = |1/2>;
		spins(electron2) = |1/2>;
		spin(nucleus1) = |1/2>;
	}

	State TPlusDOWN
	{
		spins(electron1) = |1/2>;
		spins(electron2) = |1/2>;
		spin(nucleus1) = |-1/2>;
	}

	State TMinusUP
	{
		spins(electron1) = |-1/2>;
		spins(electron2) = |-1/2>;
		spin(nucleus1) = |1/2>;
	}

	State TMinusDOWN
	{
		spins(electron1) = |-1/2>;
		spins(electron2) = |-1/2>;
		spin(nucleus1) = |-1/2>;
	}
	
	State Identity
	{
	}

	State Singlet { spins(electron1, electron2) = |1/2,-1/2> - |-1/2,1/2>; }
	State T0 { spins(electron1, electron2) = |1/2,-1/2> + |-1/2,1/2>; }
	State TP { spins(electron1) = |1/2>; spins(electron2) = |1/2>;}
	State TD { spins(electron1) = |-1/2>; spins(electron2) = |-1/2>;}	

	Transition SingletDecay
	{
		rate = 10;
		source = Singlet;
	}
	
	Transition spinindependent_decay
	{
		rate = 1;
		source = Identity;
	}

	Transition RPtransitionSingletUP 	{rate = 10; source = SingletUP; targetsystem = RPD; targetstate = SingletUP;}
	Transition RPtransitionSingletDOWN 	{rate = 10; source = SingletDOWN; targetsystem = RPD; targetstate = SingletDOWN;}
	Transition RPtransitionTripletUP 	{rate = 10; source = T0UP; targetsystem = RPD; targetstate = T0UP;}
	Transition RPtransitionTripletDOWN 	{rate = 10; source = T0DOWN; targetsystem = RPD; targetstate = T0DOWN;}
	Transition RPtransitionTPlusUP 	{rate = 10; source = TPlusUP; targetsystem = RPD; targetstate = TPlusUP;}
	Transition RPtransitionTPlusDOWN 	{rate = 10; source = TPlusDOWN; targetsystem = RPD; targetstate = TPlusDOWN;}
	Transition RPtransitionTMinusUP 	{rate = 10; source = TMinusUP; targetsystem = RPD; targetstate = TMinusDOWN;}
	Transition RPtransitionTMinusDOWN 	{rate = 10; source = TMinusUP; targetsystem = RPD; targetstate = TMinusDOWN;}

	Properties prop
	{
		initialstate = Singlet;
	}	
		
}

SpinSystem RPD
{
	Spin electron1
	{
		spin = 1/2;
		type = electron;
		tensor = isotropic(2);
	}

	Spin electron2
	{
		spin = 1/2;
		type = electron;
		tensor = isotropic(2);
	}

	Spin nucleus1
	{
		spin = 1/2;
		type = nucleus;
		tensor = isotropic("1.0");
	}
	
	Interaction Zeeman
	{
		type = Zeeman;
		field = "0 0 5e-5";
		spins = electron1, electron2;
	}

	Interaction Hyperfine3
	{
		type = Hyperfine;
		group1 = electron1;
		group2 = nucleus1;
		tensor=matrix(" 0.98491908 3.28010265 -0.53784491;
						3.28010265 25.88547678 -1.6335986;
						-0.53784491 -1.6335986 1.41368001 ");
		prefactor =3.568245455e-5;
	}

	State SingletUP
	{
		spins(electron1, electron2) = |1/2,-1/2> - |-1/2,1/2>;
		spin(nucleus1) = |1/2>;
	}

	State SingletDOWN
	{
		spins(electron1, electron2) = |1/2,-1/2> - |-1/2,1/2>;
		spin(nucleus1) = |-1/2>;
	}

	State T0UP
	{
		spins(electron1, electron2) = |1/2,-1/2> + |-1/2,1/2>;
		spin(nucleus1) = |1/2>;
	}
	State T0DOWN
	{
		spins(electron1, electron2) = |1/2,-1/2> + |-1/2,1/2>;
		spin(nucleus1) = |-1/2>;
	}

	State TPlusUP
	{
		spins(electron1) = |1/2>;
		spins(electron2) = |1/2>;
		spin(nucleus1) = |1/2>;
	}

	State TPlusDOWN
	{
		spins(electron1) = |1/2>;
		spins(electron2) = |1/2>;
		spin(nucleus1) = |-1/2>;
	}

	State TMinusUP
	{
		spins(electron1) = |-1/2>;
		spins(electron2) = |-1/2>;
		spin(nucleus1) = |1/2>;
	}

	State TMinusDOWN
	{
		spins(electron1) = |-1/2>;
		spins(electron2) = |-1/2>;
		spin(nucleus1) = |-1/2>;
	}
	
	State Identity
	{
	}

	State Singlet { spins(electron1, electron2) = |1/2,-1/2> - |-1/2,1/2>; }
	State T0 { spins(electron1, electron2) = |1/2,-1/2> + |-1/2,1/2>; }
	State TP { spins(electron1) = |1/2>; spins(electron2) = |1/2>;}
	State TD { spins(electron1) = |-1/2>; spins(electron2) = |-1/2>;}	

	Transition RPtransitionSingletUP 	{rate = 10; source = SingletUP; targetsystem = RPC; targetstate = SingletUP;}
	Transition RPtransitionSingletDOWN 	{rate = 10; source = SingletDOWN; targetsystem = RPC; targetstate = SingletDOWN;}
	Transition RPtransitionTripletUP 	{rate = 10; source = T0UP; targetsystem = RPC; targetstate = T0UP;}
	Transition RPtransitionTripletDOWN 	{rate = 10; source = T0DOWN; targetsystem = RPC; targetstate = T0DOWN;}
	Transition RPtransitionTPlusUP 	{rate = 10; source = TPlusUP; targetsystem = RPC; targetstate = TPlusUP;}
	Transition RPtransitionTPlusDOWN 	{rate = 10; source = TPlusDOWN; targetsystem = RPC; targetstate = TPlusDOWN;}
	Transition RPtransitionTMinusUP 	{rate = 10; source = TMinusUP; targetsystem = RPC; targetstate = TMinusDOWN;}
	Transition RPtransitionTMinusDOWN 	{rate = 10; source = TMinusUP; targetsystem = RPC; targetstate = TMinusDOWN;}


	Transition spinindependent_decay2
	{
		rate = 1;
		source = Identity;
	}

	Properties prop
	{
		initialstate = zero;
	}

}

Run
{
	Task CalculateQuantumYeild
	{
		type = MultiStaticSS-timeevolution; 
		//MultiStaticSS
		logfile = "logfile.log";
		datafile = "result.dat";
		timestep = 1e-4;
		totaltime = 12;
		propagator = exp; 
		//RK45
	}
}

Settings
{
	Settings general
	{
		steps = 1;
	}
}



