movie <- 3706;
user <- 6040;

#movie <- 37;
#user <- 60;
Yomega <- matrix(data=NA, nrow=user,ncol=movie);

train_data <- read.table("r3train");

for (i in 1:length(train_data[,1]))
  Yomega[train_data[i,1],train_data[i,2]] <- train_data[i,3];


YomegaC <- as(Yomega,"Incomplete")
fit <- softImpute(Yomega, rank.max=40, lambda=1, maxit = 1000, trace.it=TRUE, type="als")

ximp <- complete(Yomega,fit)


test_data <- read.table("r3test");

error <- 0;

for (i in 1:length(test_data[,1]))
  error  <- error + (ximp[test_data[i,1],test_data[i,2]] - test_data[i,3])^2;
