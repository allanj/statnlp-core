set -e -x
python get_text.py ../geoquery-new/geoquery+ID+FA+SV-UTF8.xml en > en.txt
for lang in de el th zh id sv fa; do
orig=../geoquery/geoFunql-$lang.corpus
grep "^nl:" $orig | sed 's/nl://' > $lang.txt
done

python synonimize.py

for lang in en de el th zh id sv fa; do
orig=../geoquery/geoFunql-$lang.corpus
python txt2crp.py $lang-syn.txt $orig > geoFunql-$lang-syn.corpus
done
