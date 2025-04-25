import { useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { setupPassword } from '@/utils/auth';
import { useToast } from '@/hooks/use-toast';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

const SetupPassword = () => {
  const [searchParams] = useSearchParams();
  const token = searchParams.get('token') || '';
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!password || !confirm) {
      toast({ title: 'Missing Information', description: 'Please enter and confirm your password.', variant: 'destructive' });
      return;
    }
    if (password !== confirm) {
      toast({ title: 'Passwords Do Not Match', description: 'Please make sure both passwords match.', variant: 'destructive' });
      return;
    }
    setIsLoading(true);
    const result = await setupPassword(token, password);
    setIsLoading(false);
    if (result.success) {
      toast({ title: 'Success', description: result.message, variant: 'default' });
      navigate('/login');
    } else {
      toast({ title: 'Error', description: result.message, variant: 'destructive' });
    }
  };

  return (
    <div className="flex flex-col min-h-screen">
      <Navbar />
      <div className="flex-grow flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-md w-full space-y-8 bg-white p-8 rounded-xl shadow-md">
          <div>
            <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
              Set Up Your Password
            </h2>
            <p className="mt-2 text-center text-sm text-gray-600">
              Please create a password for your account.
            </p>
          </div>
          <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
            <div className="space-y-4">
              <div>
                <Label htmlFor="password" className="block text-sm font-medium text-gray-700">
                  New Password
                </Label>
                <Input
                  id="password"
                  name="password"
                  type="password"
                  autoComplete="new-password"
                  required
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  className="mt-1"
                  placeholder="Enter new password"
                />
              </div>
              <div>
                <Label htmlFor="confirm" className="block text-sm font-medium text-gray-700">
                  Confirm Password
                </Label>
                <Input
                  id="confirm"
                  name="confirm"
                  type="password"
                  autoComplete="new-password"
                  required
                  value={confirm}
                  onChange={e => setConfirm(e.target.value)}
                  className="mt-1"
                  placeholder="Re-enter password"
                />
              </div>
            </div>
            <div>
              <Button
                type="submit"
                className="w-full bg-terracotta hover:bg-terracotta/90"
                disabled={isLoading}
              >
                {isLoading ? 'Setting Password...' : 'Set Password'}
              </Button>
            </div>
          </form>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default SetupPassword;
