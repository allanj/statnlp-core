repository_name=statnlp_optimization
git config --global credential.helper cache
git init
git add *
git commit -a -m "optimization works through copy"
git remote add $repository_name https://gitlab.com/sutd_nlp/statnlp-core.git
git remote -v
git push $repository_name optimization_1

