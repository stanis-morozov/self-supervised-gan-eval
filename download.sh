#!/bin/bash

wget https://www.dropbox.com/s/b07nqh8eu68xe2r/human_eval_images.tar.gz.0001?dl=1 -O human_eval_images.tar.gz.0001

wget https://www.dropbox.com/s/nul6c442x5gectc/human_eval_images.tar.gz.0002?dl=1 -O human_eval_images.tar.gz.0002
cat human_eval_images.tar.gz.0002 >> human_eval_images.tar.gz.0001
rm human_eval_images.tar.gz.0002

wget https://www.dropbox.com/s/2teiiy1hn88kq3u/human_eval_images.tar.gz.0003?dl=1 -O human_eval_images.tar.gz.0003
cat human_eval_images.tar.gz.0003 >> human_eval_images.tar.gz.0001
rm human_eval_images.tar.gz.0003

wget https://www.dropbox.com/s/5v1hohwvj8d8xip/human_eval_images.tar.gz.0004?dl=1 -O human_eval_images.tar.gz.0004
cat human_eval_images.tar.gz.0004 >> human_eval_images.tar.gz.0001
rm human_eval_images.tar.gz.0004

wget https://www.dropbox.com/s/0xkgiwpj603tlxc/human_eval_images.tar.gz.0005?dl=1 -O human_eval_images.tar.gz.0005
cat human_eval_images.tar.gz.0005 >> human_eval_images.tar.gz.0001
rm human_eval_images.tar.gz.0005

wget https://www.dropbox.com/s/g7y75p45394spaa/human_eval_images.tar.gz.0006?dl=1 -O human_eval_images.tar.gz.0006
cat human_eval_images.tar.gz.0006 >> human_eval_images.tar.gz.0001
rm human_eval_images.tar.gz.0006

wget https://www.dropbox.com/s/b4s82ngrq6rq7xd/human_eval_images.tar.gz.0007?dl=1 -O human_eval_images.tar.gz.0007
cat human_eval_images.tar.gz.0007 >> human_eval_images.tar.gz.0001
rm human_eval_images.tar.gz.0007

wget https://www.dropbox.com/s/8aomwcn6rhviry8/human_eval_images.tar.gz.0008?dl=1 -O human_eval_images.tar.gz.0008
cat human_eval_images.tar.gz.0008 >> human_eval_images.tar.gz.0001
rm human_eval_images.tar.gz.0008

wget https://www.dropbox.com/s/8o50b2wwdwt3pm0/human_eval_images.tar.gz.0009?dl=1 -O human_eval_images.tar.gz.0009
cat human_eval_images.tar.gz.0009 >> human_eval_images.tar.gz.0001
rm human_eval_images.tar.gz.0009

mv human_eval_images.tar.gz.0001 human_eval_images.tar.gz
