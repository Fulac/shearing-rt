files = system("ls n*t*.dat")
print( files )

set size square
set terminal png size 1100,900 lw 5 font ",28"
set xl "x"
set yl "y"
set palette defined (-1 "blue", 0 "white", 1 "red")

do for [file in files]{
    # balance out the range for the data column
    stats file u 3 nooutput # using n throws out all data outside of yrange for column n
    range = abs(STATS_max) > abs(STATS_min) ? abs(STATS_max) : abs(STATS_min)
    if( range == 0 ){
        range = 1.0
    }
    set cbrange [-range:+range]

    print sprintf( "plotting ".file." for t = ".file[9:17]."... [range = %f]", range )
    set title "t = ".file[9:17]
    set output "img-".file[9:17].".png"

    plot [0:pi][0:pi] file using 1:2:3 with image t ""
}
