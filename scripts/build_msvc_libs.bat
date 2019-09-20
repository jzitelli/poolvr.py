set OLDDIR=%CD%
cd %~dp0..\poolvr\physics
"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx86\x86\lib.exe" /def:.\collisions.def /machine:x64 /subsystem:posix /verbose
chdir /d %OLDDIR%
