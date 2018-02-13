#!/bin/sh

wget -O dexnet_2_database.hdf5 https://berkeley.box.com/shared/static/whemgcmik55qoriggm25g427kh8m2q0q.hdf5
wget -O database_direct_link.md5 https://berkeley.box.com/shared/static/spmpyn8tuu8gvvnnes8ci0cychpil2xw.md5
md5sum -c database_direct_link.md5
