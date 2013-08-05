initJTK <- function(numPoints, periods, interval) {
	source("~/JTK_Cycle (Sept 16, 2011)/JTK_CYCLE.R")
	options(stringsAsFactors=FALSE)
	jtkdist(numPoints)
	jtk.init(periods,interval)
}

runJTK <- function(data) {
	res <- apply(data,1,function(z) {
		jtkx(z)
		c(JTK.ADJP,JTK.PERIOD,JTK.LAG,JTK.AMP)
	})
	res <- as.data.frame(t(res))
	bhq <- p.adjust(unlist(res[,1]),"BH")
	res <- cbind(bhq,res)
	colnames(res) <- c("BH.Q","ADJ.P","PER","LAG","AMP")
	return(res)
}
