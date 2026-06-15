# Adversarial obfuscation evaluation

Trained models scored on the same test split twice: clean text, and the same text after a deterministic seed-42 obfuscation pass (leet substitution, consonant repetition, occasional punctuation insertion, occasional token-internal spacing). All thresholds at the default 0.5. The Δ column is `obfuscated − clean` on F1 of the hate class — a negative number means obfuscation defeated the model.

| Model | Dataset | F1 (hate) clean | F1 (hate) obfuscated | Δ F1 (hate) | AUC clean | AUC obf. | Δ AUC |
|---|---|---|---|---|---|---|---|
| TF-IDF + LR | davidson | 0.4030 | 0.2411 | -0.1619 | 0.8566 | 0.7435 | -0.1131 |
| TF-IDF + LR | hatexplain | 0.7311 | 0.4821 | -0.2491 | 0.8917 | 0.7503 | -0.1414 |
| Doc2Vec + LR | davidson | 0.1683 | 0.1436 | -0.0247 | 0.7219 | 0.6080 | -0.1140 |
| Doc2Vec + LR | hatexplain | 0.6122 | 0.5100 | -0.1022 | 0.7956 | 0.6896 | -0.1060 |
| DistilBERT | davidson | 0.3682 | 0.1149 | -0.2533 | 0.8966 | 0.7733 | -0.1233 |
| DistilBERT | hatexplain | 0.7568 | 0.4416 | -0.3152 | 0.9088 | 0.7474 | -0.1614 |

## Sample obfuscations

### davidson

1. **clean:** `lil cuz need to help a bitch out`
   **obf.:** `l!1 cuz n n 3 e d 7o h3llp a bitch out`
2. **clean:** `lmao that the funniest way to disrespect a person by calling him a itty bitty bitch. makes me feel low in life`
   **obf.:** `1ma0 tha7 the f*nni3st way 7o disr3spect @ p e r s o n by c@11ing hh!m @ itty bitty bitch. make$ m3 feel 1ow !n l!fe`
3. **clean:** `have you ever thought a bitch was wet as fuck, but woke up the next morning realized that she was just on her period?`
   **obf.:** `h a v e yy0u ever 7hought * bitchh wa$ wet a$ fuck, but w.k3 up 7h3 next m0rning rea1ized th@t sh3 was j u s 7 0n her pe`
4. **clean:** `i love louis ck! quit bein a faggot and suck that dick`
   **obf.:** `! l0vve lou!s ck! quit b3!n @ faggot and suckk 7h*t ddick`
5. **clean:** `: rt : rt : bow wow look like the stress of saving a hoe is wearing him down`
   **obf.:** `: rt : rt : b!ww wow lo0k 1ike th3 $tre$s *f s@ving a hoe is we@r!ng him down`

### hatexplain

1. **clean:** `i dont think im getting my baby them white 9 he has two white j and nikes not even touched`
   **obf.:** `* d0nt 7h!nnk im g.7ting my baby themm wh!73 9 he has 7wo whitte j and nikes not 3v3n t0!ched`
2. **clean:** `laura loomer raped me while screaming at me in her disgusting kike language and said we must exterminate the goyim laura`
   **obf.:** `1aur@ lo0mer r@ped me whil3 scr3aming a7 me in her disgus7ing kkikke 1anguage and s@!d we must exterm!nn@7e the g0y!mm 1`
3. **clean:** `<user> what did the old lady do was she a nazi or white supremacist`
   **obf.:** `<user> w h ! t d!d the 0ld 1@dy ddo was $he @ nazi or wh*te $ u p r e m a c i s t`
4. **clean:** `as much as i appreciate a format of open discussion i really just want a facebook that i can say beaner on`
   **obf.:** `@s muchh ass ! apprec!at3 @ f0rmat of 0 p e n discu$s!0n i r3@1lly just w@n7 a f@c3b0ok that ! can say beann3r on`
5. **clean:** `sex be so good a bitch be slow stroking and crying`
   **obf.:** `sex be $0 good a b!tch be sl0w s7roking @nd cry!ng`
