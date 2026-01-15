// CW EPR powder-average test: axial g-tensor in rotating-frame approximation
SpinSystem system1
{
    Spin E
    {
        type = electron;
        spin = 1/2;
        tensor = matrix("2.0 0 0; 0 2.0 0; 0 0 2.4");
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
        steps = 41;
        notifications = details;
    }

    Action sweep
    {
        type = AddVector;
        vector = system1.zeeman.field;
        direction = "0 0 1";
        value = 0.0002;
    }
}

Run
{
    Task cw_epr_powder
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
        powdersamplingpoints = 30;
        hamiltonianh0list = zeeman;
        logfile = "cw_epr_powder_g_axial.log";
        datafile = "cw_epr_powder_g_axial.dat";
        appenddata = true;
        appendlog = true;
    }
}
