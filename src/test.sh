
echo -e "14\n15\n16\n18\n19\n20" | gshuf > tmp 
a=`head -n 4 tmp`
echo $a

