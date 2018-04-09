# -*- coding:utf-8 -*-
import sys
import logging
import unittest
import os

reload(sys)
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + r'\..')  # 返回脚本的路径
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='log_test.log',
                    filemode='w')
logger = logging.getLogger()


class SomeTest(unittest.TestCase):
    def testSomething(self):
        logger.debug("this= %r", 'aaa')
        logger.debug("that= %r", 'bbb')
        # etc.
        self.assertEquals(3.14, 3.14, 'nonono')

if __name__ == "__main__":
    unittest.main()
