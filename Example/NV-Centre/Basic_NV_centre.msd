SpinSystem GroundState
{
    Spin e1
    {
        spin = 1/2;
        type = electron;
        tensor = isotropic(2);
    }

    Spin e2
    {
        spin = 1/2;
        type = electron;
        tensor = isotropic(2);
    }

    //nitrogen-14
    Spin N14
    { 
        spin = 1;
        type = nucleus;
        tensor = isotropic("1.0");
    }


//Interactions

    Interaction Dipole
    {
        type = dipole;
        group1 = e1;
        group2 = e2;
        tensor = anisotropic("-1.91 -1.91 3.82"); //GHz
        prefactor = 0.07136498013;
        IgnoreTensors = true;
    }

    Interaction E1N14
    {
        type = hyperfine;
        group1 = e1;
        group2 = N14;
        tensor = anisotropic("-2.70 -2.70 -2.14); //Mhz
        prefactor = 0.035682426404996e-3;
        //prefactor = 0.035682426404996e-2;
    }

    Interaction E2N14
    {
        type = hyperfine;
        group1 = e2;
        group2 = N14;
        tensor = anisotropic("-2.70 -2.70 -2.14); //Mhz
        prefactor = 0.035682426404996e-3;
        //prefactor = 0.035682426404996e-2;
    }

    Interaction N14nqp
    {
        type = zfs;
        group1 = N14;
        dvalue = -5.01;
        evalue = 0.0;
        prefactor = 1.098208592e-7;
        IgnoreTensors = true;
    }

    Interaction zeeman
    {
        type = singlespin;
        spins = e1,e2;
        field = "0.0 0.0 0.0";
        //field = "0.0 0.0 38e-3";
    }

    Interaction nuclearZeeman
    {
        type = singlespin;
        spins = N14;
        field = "0.0 0.0 0.0";
        //field = "0.0 0.0 38e-3";
        prefactor = 3.077724087e-3; //0.4037 * nuclear_magneton
        commonprefactor = false;
    }

//States

    State S     {spins(e1,e2) = |1/2,-1/2> - |-1/2,1/2>;} //potentially not needed
    State T0    {spins(e1,e2) = |1/2,-1/2> + |-1/2,1/2>;}
    State TP    {spin(e1) = |1/2>; spin(e2) = |1/2>;}
    State TD    {spin(e1) = |-1/2>; spin(e2) = |-1/2>;}

    //consider all possible nuclear spin configurations

    State S_U   {spins(e1,e2) = |1/2,-1/2> - |-1/2,1/2>; spin(N14) = |1>;}
    State S_Z   {spins(e1,e2) = |1/2,-1/2> - |-1/2,1/2>; spin(N14) = |0>;}
    State S_D   {spins(e1,e2) = |1/2,-1/2> - |-1/2,1/2>; spin(N14) = |-1>;}

    State T0_U  {spins(e1,e2) = |1/2,-1/2> + |-1/2,1/2>; spin(N14) = |1>;}
    State T0_Z  {spins(e1,e2) = |1/2,-1/2> + |-1/2,1/2>; spin(N14) = |0>;}
    State T0_D  {spins(e1,e2) = |1/2,-1/2> + |-1/2,1/2>; spin(N14) = |-1>;}

    State TP_U    {spin(e1) = |1/2>; spin(e2) = |1/2>; spin(N14) = |1>;}
    State TP_Z    {spin(e1) = |1/2>; spin(e2) = |1/2>; spin(N14) = |0>;}
    State TP_D    {spin(e1) = |1/2>; spin(e2) = |1/2>; spin(N14) = |-1>;}

    State TD_U    {spin(e1) = |-1/2>; spin(e2) = |-1/2>; spin(N14) = |1>;}
    State TD_Z    {spin(e1) = |-1/2>; spin(e2) = |-1/2>; spin(N14) = |0>;}
    State TD_D    {spin(e1) = |-1/2>; spin(e2) = |-1/2>; spin(N14) = |-1>;}

    State Identity
    {

    }

    Properties prop
	{
		initialstate = T0;
	}

//Transitions - OpticalPumping

    Transition TO_U_Pumping {type = sink; source = T0_U; targetsystem = ExcitedState; targetstate = T0_U; rate = 10e-3; }
    Transition TO_Z_Pumping {type = sink; source = T0_Z; targetsystem = ExcitedState; targetstate = T0_Z; rate = 10e-3; }
    Transition TO_D_Pumping {type = sink; source = T0_D; targetsystem = ExcitedState; targetstate = T0_D; rate = 10e-3; }
    Transition TP_U_Pumping {type = sink; source = TP_U; targetsystem = ExcitedState; targetstate = TP_U; rate = 10e-3; }
    Transition TP_Z_Pumping {type = sink; source = TP_Z; targetsystem = ExcitedState; targetstate = TP_Z; rate = 10e-3; }
    Transition TP_D_Pumping {type = sink; source = TP_D; targetsystem = ExcitedState; targetstate = TP_D; rate = 10e-3; }
    Transition TD_U_Pumping {type = sink; source = TD_U; targetsystem = ExcitedState; targetstate = TD_U; rate = 10e-3; }
    Transition TD_Z_Pumping {type = sink; source = TD_Z; targetsystem = ExcitedState; targetstate = TD_Z; rate = 10e-3; }
    Transition TD_D_Pumping {type = sink; source = TD_D; targetsystem = ExcitedState; targetstate = TD_D; rate = 10e-3; }
}

SpinSystem ExcitedState
{
    Spin e1
    {
        spin = 1/2;
        type = electron;
        tensor = isotropic(2);
    }

    Spin e2
    {
        spin = 1/2;
        type = electron;
        tensor = isotropic(2);
    }

    //nitrogen-14
    Spin N14
    { 
        spin = 1;
        type = nucleus;
        tensor = isotropic("1.0");
    }

//Interactions

    Interaction Dipole
    {
        type = dipole;
        group1 = e1;
        group2 = e2;
        tensor = anisotropic("-0.946 -0.946 1.893"); //GHz
        prefactor = 0.07136498013;
        IgnoreTensors = true;
    }

    Interaction E1N14
    {
        type = hyperfine;
        group1 = e1;
        group2 = N14;
        tensor = anisotropic("-2.10 -2.10 -2.30); //Mhz
        prefactor = 0.035682426404996e-3;
        //prefactor = 0.035682426404996e-2;

    }

    Interaction E2N14
    {
        type = hyperfine;
        group1 = e2;
        group2 = N14;
        tensor = anisotropic("-2.10 -2.10 -2.30); //Mhz
        prefactor = 0.035682426404996e-3;
        //prefactor = 0.035682426404996e-2;
    }

    Interaction N14nqp
    {
        type = zfs;
        group1 = N14;
        D = -5.01;
        E = 0.0;
        prefactor = 1.098208592e-7;
        IgnoreTensors = true;
    }

    Interaction zeeman
    {
        type = singlespin;
        spins = e1,e2;
        field = "0.0 0.0 0.0";
        //field = "0.0 0.0 38e-3";
    }

    Interaction nuclearZeeman
    {
        type = singlespin;
        spins = N14;
        field = "0.0 0.0 0.0";
        //field = "0.0 0.0 38e-3";
        prefactor = 3.077724087e-3; //0.4037 * nuclear_magneton
        commonprefactor = false;
    }

//States

    State S     {spins(e1,e2) = |1/2,-1/2> - |-1/2,1/2>;} //potentially not needed
    State T0    {spins(e1,e2) = |1/2,-1/2> + |-1/2,1/2>;}
    State TP    {spin(e1) = |1/2>; spin(e2) = |1/2>;}
    State TD    {spin(e1) = |-1/2>; spin(e2) = |-1/2>;}

    //consider all possible nuclear spin configurations

    State S_U   {spins(e1,e2) = |1/2,-1/2> - |-1/2,1/2>; spin(N14) = |1>;}
    State S_Z   {spins(e1,e2) = |1/2,-1/2> - |-1/2,1/2>; spin(N14) = |0>;}
    State S_D   {spins(e1,e2) = |1/2,-1/2> - |-1/2,1/2>; spin(N14) = |-1>;}

    State T0_U  {spins(e1,e2) = |1/2,-1/2> + |-1/2,1/2>; spin(N14) = |1>;}
    State T0_Z  {spins(e1,e2) = |1/2,-1/2> + |-1/2,1/2>; spin(N14) = |0>;}
    State T0_D  {spins(e1,e2) = |1/2,-1/2> + |-1/2,1/2>; spin(N14) = |-1>;}

    State TP_U    {spin(e1) = |1/2>; spin(e2) = |1/2>; spin(N14) = |1>;}
    State TP_Z    {spin(e1) = |1/2>; spin(e2) = |1/2>; spin(N14) = |0>;}
    State TP_D    {spin(e1) = |1/2>; spin(e2) = |1/2>; spin(N14) = |-1>;}

    State TD_U    {spin(e1) = |-1/2>; spin(e2) = |-1/2>; spin(N14) = |1>;}
    State TD_Z    {spin(e1) = |-1/2>; spin(e2) = |-1/2>; spin(N14) = |0>;}
    State TD_D    {spin(e1) = |-1/2>; spin(e2) = |-1/2>; spin(N14) = |-1>;}

    State Identity
    {

    }

    Properties prop
	{
		initialstate = zero;
	}
//Transitions
    //radiative-decay

    Transition TO_U_Decay {type = sink; source = T0_U; targetsystem = GroundState; targetstate = T0_U; rate = 80e-3; }
    Transition TO_Z_Decay {type = sink; source = T0_Z; targetsystem = GroundState; targetstate = T0_Z; rate = 80e-3; }
    Transition TO_D_Decay {type = sink; source = T0_D; targetsystem = GroundState; targetstate = T0_D; rate = 80e-3; }
    Transition TP_U_Decay {type = sink; source = TP_U; targetsystem = GroundState; targetstate = TP_U; rate = 80e-3; }
    Transition TP_Z_Decay {type = sink; source = TP_Z; targetsystem = GroundState; targetstate = TP_Z; rate = 80e-3; }
    Transition TP_D_Decay {type = sink; source = TP_D; targetsystem = GroundState; targetstate = TP_D; rate = 80e-3; }
    Transition TD_U_Decay {type = sink; source = TD_U; targetsystem = GroundState; targetstate = TD_U; rate = 80e-3; }
    Transition TD_Z_Decay {type = sink; source = TD_Z; targetsystem = GroundState; targetstate = TD_Z; rate = 80e-3; }
    Transition TD_D_Decay {type = sink; source = TD_D; targetsystem = GroundState; targetstate = TD_D; rate = 80e-3; }

    //isc
    Transition TO_U_ISC {type = sink; source = T0_U; targetsystem = Singlet; targetstate = S_U; rate = 2e-3; }
    Transition TO_Z_ISC {type = sink; source = T0_Z; targetsystem = Singlet; targetstate = S_Z; rate = 2e-3; }
    Transition TO_D_ISC {type = sink; source = T0_D; targetsystem = Singlet; targetstate = S_D; rate = 2e-3; }
    Transition TP_U_ISC {type = sink; source = TP_U; targetsystem = Singlet; targetstate = S_U; rate = 40e-3; }
    Transition TP_Z_ISC {type = sink; source = TP_Z; targetsystem = Singlet; targetstate = S_Z; rate = 40e-3; }
    Transition TP_D_ISC {type = sink; source = TP_D; targetsystem = Singlet; targetstate = S_D; rate = 40e-3; }
    Transition TD_U_ISC {type = sink; source = TD_U; targetsystem = Singlet; targetstate = S_U; rate = 40e-3; }
    Transition TD_Z_ISC {type = sink; source = TD_Z; targetsystem = Singlet; targetstate = S_Z; rate = 40e-3; }
    Transition TD_D_ISC {type = sink; source = TD_D; targetsystem = Singlet; targetstate = S_D; rate = 40e-3; }
}

SpinSystem Singlet
{
    Spin e1
    {
        spin = 1/2;
        type = electron;
        tensor = isotropic(2);
    }

    Spin e2
    {
        spin = 1/2;
        type = electron;
        tensor = isotropic(2);
    }

    //nitrogen-14
    Spin N14
    { 
        spin = 1;
        type = nucleus;
        tensor = isotropic("1.0");
    }

//states

    State S_U   {spins(e1,e2) = |1/2,-1/2> - |-1/2,1/2>; spin(N14) = |1>;}
    State S_Z   {spins(e1,e2) = |1/2,-1/2> - |-1/2,1/2>; spin(N14) = |0>;}
    State S_D   {spins(e1,e2) = |1/2,-1/2> - |-1/2,1/2>; spin(N14) = |-1>;}

    State Identity
    {

    }

//Transitions  
    //isc-radiative
    Transition SU_T0_ISC {type = sink; source = S_U; targetsystem = GroundState; targetstate = T0_U; rate = 5e-3;}
    Transition SZ_T0_ISC {type = sink; source = S_Z; targetsystem = GroundState; targetstate = T0_Z; rate = 5e-3;}
    Transition SD_T0_ISC {type = sink; source = S_D; targetsystem = GroundState; targetstate = T0_D; rate = 5e-3;}
    Transition SU_TP_ISC {type = sink; source = S_U; targetsystem = GroundState; targetstate = TP_U; rate = 3e-3;}
    Transition SZ_TP_ISC {type = sink; source = S_Z; targetsystem = GroundState; targetstate = TP_Z; rate = 3e-3;}
    Transition SD_TP_ISC {type = sink; source = S_D; targetsystem = GroundState; targetstate = TP_D; rate = 3e-3;}
    Transition SU_TD_ISC {type = sink; source = S_U; targetsystem = GroundState; targetstate = TD_U; rate = 3e-3;}
    Transition SZ_TD_ISC {type = sink; source = S_Z; targetsystem = GroundState; targetstate = TD_Z; rate = 3e-3;}
    Transition SD_TD_ISC {type = sink; source = S_D; targetsystem = GroundState; targetstate = TD_D; rate = 3e-3;}

    Properties prop
    {
        initialstate = zero;
    }
}

Run
{
    //Task CalculateQuantumYeild
    //{  
    //	type = MultiStaticSS-timeevolution;    
    //	//MultiStaticSS
    //	logfile = "NVtesting/exmple_NV_centre_timeevo38.log";  
    //	datafile = "NVtesting/exmple_NV_centre_timeevo38.dat"; 
    //	timestep = 1;   
    //	totaltime = 20000;
    //	propagator = exp; 
    //    transitionyields = false;
    //	//RK45
    //}

    Task EigenValue
    {
        type = eigenvalues;
        //Hamiltonian = true;
        logfile = "eigenvaluesZ.log";
        //datafile = "../../results/eigenvaluesXXZZZ.dat";
        datafile = "eigenvaluesZ.dat";
        eigenvectors = true;
    }
}



Settings
{
	Settings general
	{
		steps = 10000;
	}

    Output fieldstrength
    {
        type = length;
        vector = GroundState.zeeman.field;
    }

    Action increasefieldstrength1
	{
		type = addvector;
		vector = GroundState.zeeman.field;
		direction = "0 0 1";
        value = 10e-5;
	}

    Action increasefieldstrength2
	{
		type = addvector;
		vector = GroundState.nuclearZeeman.field;
		direction = "0 0 1"; 
        value = 10e-5;
	}

    Action increasefieldstrength3
	{
		type = addvector;
		vector = ExcitedState.zeeman.field;
		direction = "0 0 1";
        value = 10e-5;
	}

    Action increasefieldstrength4
	{
		type = addvector;
		vector = ExcitedState.nuclearZeeman.field;
		direction = "0 0 1";
        value = 10e-5;
	}
}