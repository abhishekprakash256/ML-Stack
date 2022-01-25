import numpy as np

def im2col(mtx, block_size):
    mtx_shape = mtx.shape
    print(mtx_shape)

    sx = mtx_shape[0] - block_size[0] + 1
    print(mtx_shape[0])
    print(block_size[0])

    sy = mtx_shape[1] - block_size[1] + 1
    print(sy)



    #creating a matrix of the final size of unfolding
    result = np.empty((block_size[0] * block_size[1], sx * sy))
    print(result)
    # Moved along the line, so the first holding column (i) does not move down along the row (j)
    for i in range(sy):
        for j in range(sx):
            #main logic line
            result[:, i * sx + j] = mtx[j:j + block_size[0], i:i + block_size[1]].ravel(order='F') 
            print(mtx[j:j + block_size[0], i:i + block_size[1]])
            #ravel opens up the matrix in the axis 0.
            #print(result)
    return result
 
 
def col2im(mtx, image_size, block_size):
    p, q = block_size
    sx = image_size[0] - p + 1
    sy = image_size[1] - q + 1
    result = np.zeros(image_size)
    weight = np.zeros(image_size)  # Weight record of each cell numbers plus the repeated many times
    col = 0
    # Moved along the line, so the first holding column (i) does not move down along the row (j)
    for i in range(sy):
        for j in range(sx):
            result[j:j + p, i:i + q] += mtx[:, col].reshape(block_size, order='F')
            weight[j:j + p, i:i + q] += np.ones(block_size)
            col += 1
    return result / weight

if __name__ == '__main__':
    mtx = np.around(np.random.rand(5, 5) * 100)
    #print('Original matrix:')
    print(mtx)
 
    #a1 = im2col(mtx, (2, 3))
    #print('Im2col (block size 2x3):')
    #print(a1)
    #b1 = col2im(a1, (5, 5), (2, 3))
    #print('Col2im recovery:')
    #print(b1)
 
    a2 = im2col(mtx, (3, 3))
    #   print('Im2col (block size 3x3):')
    print(a2)
    #b2 = col2im(a2, (5, 5), (3, 3))
    #print('Col2im recovery:')
    #print(b2)