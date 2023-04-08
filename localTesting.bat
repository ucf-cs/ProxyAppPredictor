REM This script is intended to run anaysis on a Windows machine locally, on
REM pickled files generated on a more capable machine ahead of time.

md ".\output"

for /L %%A in (0,1,32) do (
    for %%B in (LAMMPS,ExaMiniMDsnap,SWFFT,nekbone,HACC-IO) do (
        for /L %%C in (0,1,1) do (
            start C:\Python310\python.exe testing.py --doML --fromCSV --modelIdx %%A --app %%B --baseline %%C --depickle > ".\output\%%A_%%B_%%C.txt"
        )
    )
)
