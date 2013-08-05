
startImage = 39;
stackArg = "open=./tmp/Original/ number=100 starting=1 increment=1 scale=100 file=[] or=[] sort" 
run("Image Sequence...", stackArg);
setSlice(startImage);
run("StackReg ", "transformation=[Rigid Body]");
run("Image Sequence... ", "format=TIFF name=Aligned start=0 digits=4 save=./tmp/Aligned/");
