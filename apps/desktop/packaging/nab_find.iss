; Inno Setup script for NAB Find (task 5.3). Build the PyInstaller folder first, then (per-user Inno Setup 6.7):
;   "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" apps\desktop\packaging\nab_find.iss
; Output: apps\desktop\dist\NAB-Find-Setup-<version>.exe
; Installs per user (no admin prompt) into %LOCALAPPDATA%\Programs\NAB Find. Uninstalling keeps the
; user's searches in %LOCALAPPDATA%\NAB Find unless they tick the box to remove them.

#define AppName "NAB Find"
#define AppVersion "0.1.0"
#define Dist "..\dist\NAB Find"

[Setup]
AppId={{9E6E73A1-B58B-4A2B-826D-A393C70EF4E0}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=NAB
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=NAB-Find-Setup-{#AppVersion}
SetupIconFile=nab_find.ico
UninstallDisplayIcon={app}\{#AppName}.exe
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=2
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "{#Dist}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppName}.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppName}.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppName}.exe"; Description: "Open {#AppName} now"; Flags: nowait postinstall skipifsilent

[Code]
// WebView2 ships with Windows 11 and current Windows 10; warn (with a link) if it is missing.
function WebView2Installed(): Boolean;
var v: String;
begin
  Result := RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', v)
    or RegQueryStringValue(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', v);
end;

function InitializeSetup(): Boolean;
var code: Integer;
begin
  Result := True;
  if not WebView2Installed() then
    if SuppressibleMsgBox('NAB Find needs the Microsoft Edge WebView2 Runtime, which is missing on this PC.' + #13#10 +
              'Open the download page now? (Install it, then run this setup again.)', mbConfirmation, MB_YESNO, IDNO) = IDYES then
      ShellExec('open', 'https://developer.microsoft.com/microsoft-edge/webview2/', '', '', SW_SHOWNORMAL, ewNoWait, code);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    // Silent uninstall (/SUPPRESSMSGBOXES) keeps the data; plain MsgBox would wait for a click even then.
    if SuppressibleMsgBox('Also delete your saved searches (thumbnails and search data)?' + #13#10 +
              'Your original video files are never touched.', mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES then
      DelTree(ExpandConstant('{localappdata}\NAB Find'), True, True, True);
end;
