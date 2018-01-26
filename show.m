function drawResult()

    test_name='log.txt.test';
    train_name='log.txt.train';

    figure(1),hold on

    [Iters,Seconds,LearningRate,loss]=textread(train_name,'%f%f%f%f','delimiter', ',','headerlines',1);
     plot(Iters,loss,'r');
    [Iters,Seconds,LearningRate,accuracy,loss]=textread(test_name,'%f%f%f%f%f','delimiter', ',','headerlines',1);
     plot(Iters,loss,'g');
     plot(Iters,accuracy,'b');

     xlabel('Iters');
     ylabel('loss/accuracy');
     legend('trianLoss','testLoss','testAccuracy');

    hold off
end