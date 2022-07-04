import ext_functions as ext
import sys

# required libraries
# nltk, nltkdata
# scikit_learn (sklearn)
# pickle

if __name__ == '__main__':

    # options
    consolidate_data = 0
    analyze_data = 0
    iterate_naive = 0
    naive_bayes_class = 0
    neural_net_class = 0
    test_document = 0

    opts = sys.argv
    for opt in opts:
        if opt == '-s':
            consolidate_data = 1
        if opt == '-a':
            analyze_data = 1
        if opt == '-nbi':
            iterate_naive = 1
            naive_bayes_class = 1
        if opt == '-nb':
            naive_bayes_class = 1
        if opt == '-nn':
            neural_net_class = 1
        if opt == '-go':
            test_document = 1

    # consolidate dataset (into data.txt)
    if consolidate_data == 1:
        ext.consolidate_files()

    # load into memory
    text_wall = ext.load_docs()
    # print(docs) # test

    # analyze some features
    if analyze_data == 1:
        ext.frequency_analysis(text_wall)

    # train classifier
    if naive_bayes_class == 1:
        ext.train_classifier_naive_bayes(text_wall, iterate_naive)

    if neural_net_class == 1:
        ext.train_classifier_multilayer_perceptron(text_wall)

    if test_document == 1:
        # Test Document
        # By Nicole Goodkind, CNN Business Updated: Sun, 19 Jun 2022 11:37:56 GMT Source: CNN Business
        new_doc = """The mission of the Federal Reserve is to foster the 
            stability of US monetary systems. It's the reason the central bank was created in 1913, and it's the reason 
            it still exists today. So when inflation threatens to potentially destabilize the dollar, it's the Fed's 
            job to spring to action. There are a number of tools at their disposal, but the most effective in this 
            situation is to cool the economy by raising interest rates. With inflation rates in the US now at 40-year 
            highs, that's what the Fed is doing. Federal Reserve chair Jerome Powell announced last week that the Fed 
            will increase interest rates by an aggressive three-quarters of a percentage point, the largest hike in 28 
            years. But he also struck a more somber tone than he had in prior meetings, admitting that some factors are 
            out of his control. The Fed's objective is to bring the inflation rate down to 2% while keeping the labor 
            market strong, said Powell said on Wednesday, but "I think that what's becoming more clear is that many 
            factors that we don't control are going to play a very significant role in deciding whether that's possible 
            or not," he said. Commodity prices, the war in Ukraine, and supply chain chaos will continue to impact 
            inflation, he said, and no change to monetary policy will mitigate those things. There is still a path to 
            lower inflation rates to 2%, he said, but that path is becoming increasingly overrun by these external 
            forces. Powell's speech was largely at odds with messaging from the White House, which has emphasized 
            that the Fed is the designated go-to inflation-fighter in the US. Earlier this month, when economic data 
            showed that inflation was still at a 40-year high and that consumer sentiment had tumbled to a record low, 
            the Biden Administration pointed to the Federal Reserve's role in getting prices under control. "The Fed 
            has the tools that it needs, and we are giving them the space that it needs to operate," said Brian Deese, 
            the director of the National Economic Council. Last week, though, Powell was pushing another narrative. 
            Those ever-increasing gas and food prices, he said, are not in his control. Appropriate monetary policy 
            alone can no longer bring us back to a 2% inflation rate with a strong labor market, he said. "So much of 
            it is really not down to monetary policy," said Powell on Wednesday. "The fallout from the war in Ukraine 
            has brought a spike in prices of energy, food, fertilizer, industrial chemicals and also just the supply 
            chains more broadly, which have been larger — or longer lasting than anticipated." Mark Zandi, chief 
            economist at Moody's Analytics, agrees with that view. "The primary culprit [of inflation] was higher
            energy prices, particularly gasoline, and a lot of that can be traced back to Russia's invasion of Ukraine 
            that caused global oil prices to spike," he said in a recent episode of his podcast, Moody's Talks. 
            Inflation should ease, when the pandemic subsides and the market adjusts to new sanctions against Russia, 
            he added. It's hard to say whether increasing interest rates will help limit the wildfire spread of 
            inflation or if it's too little too late. Powell seems to be hedging. "I think events of the last few 
            months have raised the degree of difficulty, created great challenges," Powell said. "And there's a much 
            bigger chance now that it will depend on factors that we don't control." The $5.7 billion bet against 
            Europe Some wealthy Americans like to vacation in Europe. Connecticut's richest man prefers to make 
            multi-billion dollar bets against the old world's economic future. Ray Dalio's Bridgewater Associates is 
            wagering nearly $6 billion that European stocks will fall. That makes the world's largest hedge fund the 
            world's largest short seller of Euro equities. All in all, Bridgewater has 18 active short bets against 
            European companies, including a $1 billion position against semiconductor company ASML Holding and a $752 
            million bet against oil and energy company TotalEnergies SE. This isn't Bridgewater's first rodeo. Dalio 
            hasn't been on Europe's side for a while. In 2020, Bridgewater bet $14 billion against stocks there and in 
            2018 they built a $22 billion short position against the region. Pourquoi? Bridgewater has been pretty mum 
            about its whole Euro strategy in general, but some clues have emerged from an interview Dalio gave to 
            Italian newspaper La Repubblica last week. He explained that Bridgewater is staying far away from countries 
            that are at risk of domestic strife or international war. He also said he's worried about central banks' 
            attempts to address high inflation and anticipates that economy will soon sour because of them. In short, 
            he's going short because of war in Ukraine and the European Central Banks' hawkish policy. But maybe it's 
            about the battle for world order. One thing Dalio hasn't been shy about is sharing his broader worldview. 
            In a series of LinkedIn blog posts he has explained why he thinks the US is rapidly heading toward civil 
            war and how the global world order is shifting. "The Russia-Ukraine-US-other-countries dynamic is the most 
            attention-grabbing part of the changing world order dynamic that is underway," he writes. "But it is 
            essentially just the first battle in what will be a long war for control of the world order." It could be 
            that Bridgewater, which has $151 billion-in-assets, is betting that Europe won't make it out of the war on 
            top. So far, that bet is paying off. The company has made a 26.2% gain in its flagship Pure Alpha fund this 
            year, while the S&P 500 has lost nearly 24%. The STOXX Europe 600, a broad index that measures the European 
            stock market is down about 17% year-to-date."""

        # classify new document (naive bayes)
        ext.use_trained_classifier_nb(new_doc)
        # classify new document (neural network)
        ext.use_trained_classifier_nn(new_doc)

    print('Complete')
