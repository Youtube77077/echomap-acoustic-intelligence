# EchoMap v1.0.1

**Fixes the startup crash** "Failed to resolve Python.Runtime.Loader.Initialize". Windows marks files extracted from a downloaded ZIP as coming from the internet, and .NET Framework then refuses to load the window component (pythonnet).
- EchoMap now removes that mark from its *own* bundled DLLs at startup, the same thing Explorer's "Unblock" does, and touches nothing outside the EchoMap folder.
- If the native window still can't start, EchoMap opens in a Microsoft Edge app window instead, with its own private profile. A native file picker is included in that mode.
- Startup problems are logged to `%LOCALAPPDATA%\EchoMap\logs\startup.log`.

Downloads:
- `EchoMap-v1.0.1-win64.zip`: the application.
- `EchoMap-models-v1.zip`: the models (unchanged from v1.0.0). Extract into `EchoMap\models`.
