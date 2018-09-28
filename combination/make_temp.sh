#!/bin/bash
name=license
password=bTStirBr3z-yMZ-7syDHUkh2EQdavhn3zQ7Bh8BhmFU_qbU6kREx2bfZXTUu9QPuOKgaQAKqxDeEtFejYgSBi7la0FTilrxuOAy_Dqw-zZMGhBo1IY023pHho77ASOAZyq6yG3VdedlGfi7zJAxFY1zRUll4cBg8ZkUrjQ0WbSE=IGhRwoTjL7PjtwVHOmFlHde21lHW0IW0iKX90RRKzrgJF--0ru_7r2eBhsZ-tPsB044-qoLRpQ95450wYl0vvozWvjsnUWeoqSS5U8IzUXqCtn4al4CK4Ofh1CwJqj9iJiUm8fjzRyjuNcYmxueLi6cG4PljfAwWdunC4xwY4ho=
host=rqdatad-pro.ricequant.com
port=16004
url=rqdatac://${name}:${password}@${host}:${port}
echo "export RQDATAC_CONF=$url" >> ~/.bash_profile
source ~/.bash_profile