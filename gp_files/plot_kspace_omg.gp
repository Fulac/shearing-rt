files = system("ls ks*t*.dat")
print( files )

#set size square
set terminal png size 1100,900 lw 5 font ",28"
set xl "kx"
set yl "ky"
#set palette defined (-1 "blue", 0 "white", 1 "red")

do for [file in files]{
    # balance out the range for the data column
    stats file u 3 nooutput # using n throws out all data outside of yrange for column n
    range = abs(STATS_max) > abs(STATS_min) ? abs(STATS_max) : abs(STATS_min)
    if( range == 0 ){
        range = 1.0
    }
    set cbrange [0:+range]

    print sprintf( "plotting ".file." for t = ".file[7:11]."... [range = %f]", range )
    set title "t = ".file[7:11]
    set output "img-t".file[7:11].".png"

    plot file using 1:2:3 with image t ""
}
