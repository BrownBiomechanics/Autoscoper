# Debugging Autoscoper - Tips and Tricks

Debugging Autoscoper can be a challenge, due to the separation of the GUI and the backend. This page will provide some tips and tricks for debugging Autoscoper.

```{note}
These instructions are for debugging Autoscoper on Windows using Visual Studio (2022 at the time of writing).
```

## Setting up Breakpoints for the Backend

Autoscoper's backend is written in C++ and is compiled as a library. This means you will have to do some extra work to set up breakpoints in the backend.

1) Compile Autoscoper in Debug mode.
2) Navigate to the inner build directory (e.g. `C:\Users\username\Documents\Autoscoper\Main-build\Autoscoper-build`).
3) Open the `Autoscoper.sln` file in Visual Studio.
4) Set the startup project to `INSTALL`.
5) Open the `INSTALL` project's properties.
6) Navigate to `Configuration Properties > Debugging`.
7) Set the `Command` to be `$(ProjectDir)install\bin\autoscoper.exe`. This should be the location of the installed Autoscoper executable (ie. The location of the `autoscoper.exe` file and all of the necessary DLLs).
8) From the main Visual Studio window, Navigate to `Tools > Options > Debugging`.
9) Uncheck `Enable Just My Code`.

You should now be able to set breakpoints in the backend code.
