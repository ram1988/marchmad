library(ROCR)
pairs<-read.csv(file="feats.csv")
attach(pairs)
#dataFrame<-data.frame(label1=label,alchemy_category_score1=alchemy_category_score,numberOfLinks1=numberOfLinks,avglinksize1=avglinksize,html_ratio1=html_ratio,image_ratio1=image_ratio,spelling_errors_ratio1=spelling_errors_ratio,linkwordscore1=linkwordscore)
#fit <- glm(formula=label~avglinksize+compression_ratio+embed_ratio+frameTagRatio+html_ratio+linkwordscore+numberOfLinks+numwords_in_url+parametrizedLinkRatio+spelling_errors_ratio,data=pairs,family=binomial())

size<-dim(pairs)
#pairs[[7]][13]
#for(p in pairs) {
 # print(p[1])
#}
for(i in 1:size[1])  {
 print(pairs[i])
 #fit <- glm(formula=pred~pairs[[2]][i]+pairs[[3]][i],data=pairs,family=binomial())
 #predicted <- predict(fit, type = "response")
 #print(predicted)
}



summary(fit)
newdata = data.frame(team1_wins=10,team2_wins=20)
predicted <- predict(fit,newdata, type = "response")
predicted
prob <- prediction(predicted, pred)
prob


#plot(cpi,xaxt="n",ylab="CPI",xlab="")
#lines(predict(fit))

#input<-data.frame(avglinksize=119,html_ratio=0.2983,spelling_errors_ratio=0.0873,linkwordscore=12)
#input<-data.frame(avglinksize=0,html_ratio=0.3283,spelling_errors_ratio=0.08333,linkwordscore=0)

#predict(fit,input,interval = "confidence",type="response")

roc.area(label,predicted)



tprfpr <- performance(prob, "tpr", "fpr")
tpr <- unlist(slot(tprfpr, "y.values"))
fpr <- unlist(slot(tprfpr, "x.values"))
roc <- data.frame(tpr, fpr)
ggplot(roc) + geom_line(aes(x = fpr, y = tpr)) +
  geom_abline(intercept = 0, slope = 1, colour = "gray") +    ylab("Sensitivity") + 
  xlab("1 - Specificity")