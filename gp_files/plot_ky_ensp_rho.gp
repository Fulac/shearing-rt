files = system("ls ky_ensp*.dat")
print( files )

set logscale
set xl "Ky"
set yl "En"
set terminal postscript eps enhanced color font ",25" 
#set terminal png size 800, 600 lw 5 font ",25"
set format y "%.2t{/Symbol \264}10^{%T}"

do for [file in files]{
    print sprintf( "plotting ".file." for t = ".file[10:18] )
    set title "t = ".file[10:18]
    set output "en-".file[10:18].".eps"

    plot file u 1:4 w lp lw 3 pt 13 t ""
}
