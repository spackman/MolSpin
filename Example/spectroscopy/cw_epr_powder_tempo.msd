// CW EPR powder-average: nitroxide-like (TEMPO) g/A tensor, rotating-frame detuning sweep
SpinSystem system1
{
    Spin E
    {
        type = electron;
        spin = 1/2;
        tensor = matrix("2.0093 0 0; 0 2.0061 0; 0 0 2.0022");
    }

    Spin N
    {
        type = nucleus;
        spin = 1;
        tensor = isotropic(1);
    }

    Interaction zeeman
    {
        type = zeeman;
        field = "0 0 -0.004"; // detuning field (rotating frame)
        spins = E;
        ignoretensors = false;
        commonprefactor = true;
        prefactor = 1;
    }

    Interaction hyperfine
    {
        type = hyperfine;
        group1 = E;
        group2 = N;
        tensor = matrix("0.0006 0 0; 0 0.0006 0; 0 0 0.0036"); // ~6 G, 6 G, 36 G
        ignoretensors = true;
        commonprefactor = true;
        prefactor = 2.0023;
    }

    Operator T1E { type = relaxationt1; rate = 0.05; spins = E; }
    Operator T2E { type = relaxationt2; rate = 0.1; spins = E; }

    State Up
    {
        spin(E) = |1/2>;
    }

    Properties properties
    {
        initialstate = Up;
    }

    Pulse cw
    {
        type = LongPulseStaticField;
        field = "0.0002 0 0";
        pulsetime = 200.0;
        timestep = 0.1;
        group = E;
        prefactorlist = 1,1,1;
        commonprefactorlist = true;
        ignoretensorslist = true;
    }
}

Settings
{
    Settings general
    {
        steps = 61;
        notifications = details;
    }

    Action sweep
    {
        type = AddVector;
        vector = system1.zeeman.field;
        direction = "0 0 1";
        value = 0.00015;
    }
}

Run
{
    Task cw_epr_powder_tempo
    {
        type = StaticHS-Direct-Spectra;
        method = timeevo;
        timestep = 0.1;
        totaltime = 0;
        integration = false;
        printtimeframe = pulse;
        integrationtimeframe = pulse;
        pulsesequence = ["cw 0"];
        spinlist = E;
        powdersamplingpoints = 60;
        hamiltonianh0list = zeeman,hyperfine;
        logfile = "cw_epr_powder_tempo.log";
        datafile = "cw_epr_powder_tempo.dat";
        appenddata = true;
        appendlog = true;
    }
}
