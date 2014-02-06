pairs<-read.csv(file="feats.csv")
attach(pairs)
#dataFrame<-data.frame(label1=label,alchemy_category_score1=alchemy_category_score,numberOfLinks1=numberOfLinks,avglinksize1=avglinksize,html_ratio1=html_ratio,image_ratio1=image_ratio,spelling_errors_ratio1=spelling_errors_ratio,linkwordscore1=linkwordscore)
#fit <- glm(formula=label~avglinksize+compression_ratio+embed_ratio+frameTagRatio+html_ratio+linkwordscore+numberOfLinks+numwords_in_url+parametrizedLinkRatio+spelling_errors_ratio,data=pairs,family=binomial())
fit <- glm(formula=label~frameTagRatio+linkwordscore+numberOfLinks+numwords_in_url-1,data=pairs,family=binomial())
fit

summary(fit)
plot(cpi,xaxt="n",ylab="CPI",xlab="")
lines(predict(fit))

#input<-data.frame(avglinksize=119,html_ratio=0.2983,spelling_errors_ratio=0.0873,linkwordscore=12)
#input<-data.frame(avglinksize=0,html_ratio=0.3283,spelling_errors_ratio=0.08333,linkwordscore=0)

#predict(fit,input,interval = "confidence",type="response")

round(abs(-0.939))

predicted <- predict(fit)
roc.area(label,predicted)
library(ROCR)
prob <- prediction(predicted, label)

tprfpr <- performance(prob, "tpr", "fpr")
tpr <- unlist(slot(tprfpr, "y.values"))
fpr <- unlist(slot(tprfpr, "x.values"))
roc <- data.frame(tpr, fpr)
ggplot(roc) + geom_line(aes(x = fpr, y = tpr)) +
  geom_abline(intercept = 0, slope = 1, colour = "gray") +    ylab("Sensitivity") + 
  xlab("1 - Specificity")