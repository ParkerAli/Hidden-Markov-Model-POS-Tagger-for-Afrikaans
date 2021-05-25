library(readxl)

train = read_excel("GOV-ZA.50000TrainingSet.af.pos.full.xls")
test = read_excel("GOV-ZA.5000TestSet.af.pos.full.xls")

# directories changed to prevent an overwrite
# set encoding because it isn't done automatically
write.csv(train,"train.csv", row.names = FALSE, fileEncoding = "UTF-8")
write.csv(test,"test.csv", row.names = FALSE, fileEncoding = "UTF-8")


## some quick checks
trainCheck = read.csv("train.csv")
testCheck = read.csv("test.csv")

#check dims
dim(trainCheck)
dim(testCheck)
dim(test)
dim(train)
View(read.csv("train.csv"))

# we can see blank rows have been made NA. Might need to address these in EDA
# NA row at every EOS
sum(is.na(trainCheck$Token))
sum(is.na(testCheck$Token))

is.na(test)





