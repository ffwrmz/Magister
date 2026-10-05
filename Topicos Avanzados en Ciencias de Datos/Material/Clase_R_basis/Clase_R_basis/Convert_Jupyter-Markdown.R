
jupy <- "Ejercicios_Practicos.ipynb"
markd <- "Ejercicios_Practicos.rmd"

# Convert from IPYNB to RMD
file_nb_rmd = rmarkdown:::convert_ipynb(jupy)
st_nb_rmd = xfun::file_string(file_nb_rmd)

#save RMD
fileConn <- file(markd)
writeLines(st_nb_rmd, fileConn)
close(fileConn)