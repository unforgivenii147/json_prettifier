# json_prettifier

added -m cli arg to minify
and default action without cli arg is to beautify
the cli script now can accept multiple
files/folders as input
- if no input is provided, script searches current path
for json files and beautify them with indent=2 , ensure_ascii=False options
- you can sort the keys of a json file by passing -s cli arg
- the acript updates files inplace
- if the input file is not a valid python json file
 script skips it
 
