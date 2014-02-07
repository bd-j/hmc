      subroutine leapfrog(ndim,  epsilon, q, p, grad, qprime, pprime, gradprime)
        integer, intent(in) :: ndim, epsilon
        double precision, intent(in), dimension(ndim) :: q, p, grad
        double precision, intent(out), dimension(ndim) :: qprime, pprime, gradprime
        
        pprime = p + epsilon/2. * grad
        qprime = theta + epsilon * pprime
        call model_lnprob_grad(qprime, gradprime)
        pprime = pprime + epsilon/2 * gradprime
        
      end subroutine

      subroutine trajectory(ndim, epsilon, length, q, qprime)
        integer, intent(in) :: ndim, epsilon, length
        double precision, intent(in), dimension(ndim) :: q
        double precision, dimension(ndim) :: p, pprime
        double precision, intent(out), dimension(ndim) :: qprime
        double precision :: lnP0, lnP
        integer :: k

        !initialize random momenta
        do k = 1, ndim
           p(k) = random_number(r)
        !find the first gradient and lnp
        call model_lnprob_grad(q, grad)
        call model_lnprob(q, lnP0)

        !leap 'length' steps
        do k=1,length
           call leapfrog(ndim, epsilon, q, p, grad, qprime, pprime, gradprime)
           q = qprime
           p = pprime
           grad = gradprime
        call model_lnprob(q, lnP)
        dU =  lnP0 - lnP !change in potential = negative change in lnP
        dK = 0.5 * (p * p  - p0 * p0) !change in kinetic
        call random_number(r)
        if r .lt. (dU + dK) then
           
        

      subroutine hmc_advance()
        integer
