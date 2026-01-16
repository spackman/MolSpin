SpinSystem GroundState
{
    Spin e1
    {
        spin = 1;
        type = electron;
        tensor = isotropic(2);
    }

    //nitrogen-14
    Spin N14
    { 
        spin = 1;
        type = nucleus;
        tensor = isotropic(1.0);
    }


//Interactions

    Interaction zfs
    {
        type = zfs;
        group1 = e1;
        dvalue = 2868.91;
        //D = 0;
        E = 0;
        prefactor = 0.035682426404996e-3;
        //prefactor = 0.2241992973e-3;
        commonprefactor = false;
        ignoretensors = true;
    }

    Interaction E1N14
    {
        type = hyperfine;
        group1 = e1;
        group2 = N14;
        tensor = anisotropic("-2.70 -2.70 -2.14"); //Mhz
        prefactor = 0.035682426404996e-3;
        //prefactor = 0.2241992973e-3;
        ignoretensors = true;
        commonprefactor = false;
    }

    Interaction N14nqp
    {
        type = zfs;
        group1 = N14;
        dvalue = -5.01;
        E = 0.0;
        prefactor = 0.3249174108;
        //prefactor = 1.020936271;
        //prefactor = 0.1193805208;
        commonprefactor = false;
    }

    Interaction zeeman
    {
        type = singlespin;
        spins = e1;
        field = "0.0 0.0 0.0";
        //field = "0.0 0.0 38e-3";
    }

    Interaction nuclearZeeman
    {
        type = singlespin;
        spins = N14;
        field = "0.0 0.0 0.0";
        //field = "0.0 0.0 38e-3";
        prefactor = 3.077e-3;//0.4037 * nuclear_magneton
        commonprefactor = false;
    }

//States

    //consider all possible nuclear spin configurations

    State T0      {spin(e1) = |1>;}

    State T0_U    {spin(e1) = |0>; spin(N14) = |1>;}
    State T0_Z    {spin(e1) = |0>; spin(N14) = |0>;}
    State T0_D    {spin(e1) = |0>; spin(N14) = |-1>;}

    State TP_U    {spin(e1) = |1>; spin(N14) = |1>;}
    State TP_Z    {spin(e1) = |1>; spin(N14) = |0>;}
    State TP_D    {spin(e1) = |1>; spin(N14) = |-1>;}

    State TD_U    {spin(e1) = |0>; spin(N14) = |1>;}
    State TD_Z    {spin(e1) = |0>; spin(N14) = |0>;}
    State TD_D    {spin(e1) = |0>; spin(N14) = |-1>;}

    State Identity
    {

    }

    Properties prop
	{
		initialstate = T0;
	}
}

Run
{
    Task EigenValue
    {
        type = eigenvalues;
        //Hamiltonian = true;
        logfile = "eigenvalues_gs_Z.log";
        datafile = "../../results/eigenvalues_gs_Z.dat";
        //datafile = "eigenvalues_gs_Z.dat";
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
}