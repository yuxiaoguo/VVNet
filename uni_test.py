from tests.ops_test import *
from tests.analysis_test import *
from tests.mtl_test import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    tf.test.main()
