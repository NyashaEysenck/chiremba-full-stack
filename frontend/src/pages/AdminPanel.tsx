import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/components/AuthContext';
import { useToast } from '@/hooks/use-toast';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { 
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { 
  AlertCircle, 
  UserPlus, 
  AlertTriangle, 
  RefreshCw, 
  Shield, 
  Users, 
  UserCog, 
  Loader2,
  Trash2,
  Edit,
  CheckCircle2
} from 'lucide-react';
import { mongoAPI } from '@/integrations/mongodb/client';
import { createAccount, deleteUser, assignAdminRole } from '@/utils/auth';
import { User } from '@/components/UserCard';
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

const AdminPanel = () => {
  const { isAuthenticated, userRole } = useAuth();
  const { toast } = useToast();
  const navigate = useNavigate();
  
  const [users, setUsers] = useState<User[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  
  // New user form state
  const [newUserOpen, setNewUserOpen] = useState(false);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [newUserRole, setNewUserRole] = useState<'user' | 'staff' | 'admin'>('staff');
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  // Delete confirmation state
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  
  // Promote confirmation state
  const [promoteDialogOpen, setPromoteDialogOpen] = useState(false);
  const [isPromoting, setIsPromoting] = useState(false);

  // Redirect if not authenticated or not an admin
  useEffect(() => {
    if (!isAuthenticated) {
      toast({
        title: "Authentication Required",
        description: "Please log in to access this page.",
        variant: "destructive",
      });
      navigate("/login");
    } else if (userRole !== 'admin') {
      toast({
        title: "Admin Access Only",
        description: "You do not have permission to access this page.",
        variant: "destructive",
      });
      navigate("/");
    }
  }, [isAuthenticated, userRole, navigate, toast]);

  // Fetch users on component mount
  useEffect(() => {
    if (isAuthenticated && userRole === 'admin') {
      fetchUsers();
    }
  }, [isAuthenticated, userRole]);

  // Function to fetch all users
  const fetchUsers = async () => {
    try {
      setIsLoading(true);
      setIsRefreshing(true);
      
      // Get all users from MongoDB API
      const userData = await mongoAPI.users.getAllUsers();
      setUsers(userData);
      
      toast({
        title: "Users Refreshed",
        description: `Successfully loaded ${userData.length} users`,
      });
    } catch (error: any) {
      console.error("Error fetching users:", error);
      toast({
        title: "Error",
        description: `Failed to load users: ${error.message}`,
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  };

  // Function to create a new user
  const handleCreateUser = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!name || !email || !password) {
      toast({
        title: "Missing Information",
        description: "Please fill in all fields",
        variant: "destructive",
      });
      return;
    }
    
    setIsSubmitting(true);
    
    try {
      const response = await createAccount(email, password, name, newUserRole);
      
      if (response.success) {
        toast({
          title: "Success",
          description: `${newUserRole.charAt(0).toUpperCase() + newUserRole.slice(1)} account created successfully`,
        });
        
        // Reset form and close dialog
        setName('');
        setEmail('');
        setPassword('');
        setNewUserOpen(false);
        
        // Refresh user list
        fetchUsers();
      } else {
        toast({
          title: "Error",
          description: response.message || "Failed to create account",
          variant: "destructive",
        });
      }
    } catch (error: any) {
      console.error("Error creating user:", error);
      toast({
        title: "Error",
        description: error.message || "Failed to create account",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  // Function to delete a user
  const handleDeleteUser = async () => {
    if (!selectedUser) return;
    
    setIsDeleting(true);
    
    try {
      const response = await deleteUser(selectedUser.id);
      
      if (response.success) {
        toast({
          title: "Success",
          description: "User deleted successfully",
        });
        
        // Remove user from state
        setUsers(users.filter(user => user.id !== selectedUser.id));
        
        // Close dialog
        setDeleteDialogOpen(false);
        setSelectedUser(null);
      } else {
        toast({
          title: "Error",
          description: response.message || "Failed to delete user",
          variant: "destructive",
        });
      }
    } catch (error: any) {
      console.error("Error deleting user:", error);
      toast({
        title: "Error",
        description: error.message || "Failed to delete user",
        variant: "destructive",
      });
    } finally {
      setIsDeleting(false);
    }
  };

  // Function to promote a user to admin
  const handlePromoteToAdmin = async () => {
    if (!selectedUser) return;
    
    setIsPromoting(true);
    
    try {
      const response = await assignAdminRole(selectedUser.id);
      
      if (response.success) {
        toast({
          title: "Success",
          description: "User promoted to admin successfully",
        });
        
        // Update user role in state
        setUsers(users.map(user => 
          user.id === selectedUser.id 
            ? { ...user, role: 'admin' } 
            : user
        ));
        
        // Close dialog
        setPromoteDialogOpen(false);
        setSelectedUser(null);
      } else {
        toast({
          title: "Error",
          description: response.message || "Failed to promote user",
          variant: "destructive",
        });
      }
    } catch (error: any) {
      console.error("Error promoting user:", error);
      toast({
        title: "Error",
        description: error.message || "Failed to promote user",
        variant: "destructive",
      });
    } finally {
      setIsPromoting(false);
    }
  };

  // Format date for display
  const formatDate = (dateString: string) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  // If user is not admin, show access denied
  if (userRole !== 'admin') {
    return (
      <div className="min-h-screen flex flex-col">
        <Navbar />
        <div className="flex-grow flex items-center justify-center p-6">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Access Denied</AlertTitle>
            <AlertDescription>
              You do not have permission to access the admin panel. Please contact an administrator if you believe this is an error.
            </AlertDescription>
          </Alert>
        </div>
        <Footer />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      
      <div className="flex-grow container mx-auto px-4 pt-16 pb-8">
        {/* Admin Panel Header with Refresh Button */}
        <div className="bg-white p-4 rounded-lg shadow mb-6">
          <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
            <h1 className="text-2xl font-bold text-gray-900">User Management</h1>
            <Button 
              variant="outline" 
              onClick={fetchUsers}
              disabled={isRefreshing}
              className="w-full sm:w-auto"
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
              Refresh Users
            </Button>
          </div>
        </div>
        
        {/* Tabs for user categories */}
        <Tabs defaultValue="all-users">
          <TabsList className="mb-4">
            <TabsTrigger value="all-users">
              <Users className="h-4 w-4 mr-2" />
              All Users
            </TabsTrigger>
            <TabsTrigger value="admin">
              <Shield className="h-4 w-4 mr-2" />
              Admins
            </TabsTrigger>
            <TabsTrigger value="staff">
              <UserCog className="h-4 w-4 mr-2" />
              Staff
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="all-users" className="space-y-4">
            {isLoading ? (
              <div className="flex justify-center items-center py-12">
                <Loader2 className="h-8 w-8 text-primary animate-spin" />
              </div>
            ) : users.length === 0 ? (
              <div className="text-center py-12 bg-gray-50 rounded-lg border border-gray-200">
                <p className="text-gray-600">No users found</p>
                <Button 
                  className="mt-4"
                  onClick={() => setNewUserOpen(true)}
                >
                  <UserPlus className="h-4 w-4 mr-2" />
                  Add First User
                </Button>
              </div>
            ) : (
              <>
                <div className="flex justify-end mb-4">
                  <Button 
                    onClick={() => setNewUserOpen(true)}
                    className="bg-blue-600 hover:bg-blue-700 text-white"
                  >
                    <UserPlus className="h-4 w-4 mr-2" />
                    Add New User
                  </Button>
                </div>
                <div className="bg-white rounded-lg shadow overflow-hidden">
                  <Table>
                    <TableCaption>List of all users in the system</TableCaption>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Name</TableHead>
                        <TableHead>Email</TableHead>
                        <TableHead>Role</TableHead>
                        <TableHead>Created</TableHead>
                        <TableHead className="text-right">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {users.map(user => (
                        <TableRow key={user.id}>
                          <TableCell className="font-medium">{user.name}</TableCell>
                          <TableCell>{user.email}</TableCell>
                          <TableCell>
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                              user.role === 'admin' 
                                ? 'bg-blue-100 text-blue-800' 
                                : user.role === 'staff'
                                  ? 'bg-green-100 text-green-800'
                                  : 'bg-gray-100 text-gray-800'
                            }`}>
                              {user.role.charAt(0).toUpperCase() + user.role.slice(1)}
                            </span>
                          </TableCell>
                          <TableCell>{user.createdAt ? formatDate(user.createdAt) : 'N/A'}</TableCell>
                          <TableCell className="text-right">
                            <div className="flex justify-end space-x-2">
                              {user.role !== 'admin' && (
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => {
                                    setSelectedUser(user);
                                    setPromoteDialogOpen(true);
                                  }}
                                >
                                  <Shield className="h-4 w-4 mr-1" />
                                  <span className="sr-only md:not-sr-only md:inline-block">Promote</span>
                                </Button>
                              )}
                              <Button
                                variant="destructive"
                                size="sm"
                                onClick={() => {
                                  setSelectedUser(user);
                                  setDeleteDialogOpen(true);
                                }}
                              >
                                <Trash2 className="h-4 w-4 mr-1" />
                                <span className="sr-only md:not-sr-only md:inline-block">Delete</span>
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </>
            )}
          </TabsContent>
          
          <TabsContent value="admin" className="space-y-4">
            {isLoading ? (
              <div className="flex justify-center items-center py-12">
                <Loader2 className="h-8 w-8 text-primary animate-spin" />
              </div>
            ) : users.filter(user => user.role === 'admin').length === 0 ? (
              <div className="text-center py-12 bg-gray-50 rounded-lg border border-gray-200">
                <p className="text-gray-600">No admin users found</p>
                <Button 
                  className="mt-4"
                  onClick={() => {
                    setNewUserRole('admin');
                    setNewUserOpen(true);
                  }}
                >
                  <UserPlus className="h-4 w-4 mr-2" />
                  Add Admin User
                </Button>
              </div>
            ) : (
              <>
                <div className="flex justify-end mb-4">
                  <Button 
                    onClick={() => {
                      setNewUserRole('admin');
                      setNewUserOpen(true);
                    }}
                    className="bg-blue-600 hover:bg-blue-700 text-white"
                  >
                    <UserPlus className="h-4 w-4 mr-2" />
                    Add New Admin User
                  </Button>
                </div>
                <div className="bg-white rounded-lg shadow overflow-hidden">
                  <Table>
                    <TableCaption>List of administrators</TableCaption>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Name</TableHead>
                        <TableHead>Email</TableHead>
                        <TableHead>Created</TableHead>
                        <TableHead className="text-right">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {users
                        .filter(user => user.role === 'admin')
                        .map(user => (
                          <TableRow key={user.id}>
                            <TableCell className="font-medium">{user.name}</TableCell>
                            <TableCell>{user.email}</TableCell>
                            <TableCell>{user.createdAt ? formatDate(user.createdAt) : 'N/A'}</TableCell>
                            <TableCell className="text-right">
                              <Button
                                variant="destructive"
                                size="sm"
                                onClick={() => {
                                  setSelectedUser(user);
                                  setDeleteDialogOpen(true);
                                }}
                              >
                                <Trash2 className="h-4 w-4 mr-1" />
                                <span className="sr-only md:not-sr-only md:inline-block">Delete</span>
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))}
                    </TableBody>
                  </Table>
                </div>
              </>
            )}
          </TabsContent>
          
          <TabsContent value="staff" className="space-y-4">
            {isLoading ? (
              <div className="flex justify-center items-center py-12">
                <Loader2 className="h-8 w-8 text-primary animate-spin" />
              </div>
            ) : users.filter(user => user.role === 'staff').length === 0 ? (
              <div className="text-center py-12 bg-gray-50 rounded-lg border border-gray-200">
                <p className="text-gray-600">No staff users found</p>
                <Button 
                  className="mt-4"
                  onClick={() => {
                    setNewUserRole('staff');
                    setNewUserOpen(true);
                  }}
                >
                  <UserPlus className="h-4 w-4 mr-2" />
                  Add Staff User
                </Button>
              </div>
            ) : (
              <>
                <div className="flex justify-end mb-4">
                  <Button 
                    onClick={() => {
                      setNewUserRole('staff');
                      setNewUserOpen(true);
                    }}
                    className="bg-blue-600 hover:bg-blue-700 text-white"
                  >
                    <UserPlus className="h-4 w-4 mr-2" />
                    Add New Staff User
                  </Button>
                </div>
                <div className="bg-white rounded-lg shadow overflow-hidden">
                  <Table>
                    <TableCaption>List of staff members</TableCaption>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Name</TableHead>
                        <TableHead>Email</TableHead>
                        <TableHead>Created</TableHead>
                        <TableHead className="text-right">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {users
                        .filter(user => user.role === 'staff')
                        .map(user => (
                          <TableRow key={user.id}>
                            <TableCell className="font-medium">{user.name}</TableCell>
                            <TableCell>{user.email}</TableCell>
                            <TableCell>{user.createdAt ? formatDate(user.createdAt) : 'N/A'}</TableCell>
                            <TableCell className="text-right">
                              <div className="flex justify-end space-x-2">
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => {
                                    setSelectedUser(user);
                                    setPromoteDialogOpen(true);
                                  }}
                                >
                                  <Shield className="h-4 w-4 mr-1" />
                                  <span className="sr-only md:not-sr-only md:inline-block">Promote</span>
                                </Button>
                                <Button
                                  variant="destructive"
                                  size="sm"
                                  onClick={() => {
                                    setSelectedUser(user);
                                    setDeleteDialogOpen(true);
                                  }}
                                >
                                  <Trash2 className="h-4 w-4 mr-1" />
                                  <span className="sr-only md:not-sr-only md:inline-block">Delete</span>
                                </Button>
                              </div>
                            </TableCell>
                          </TableRow>
                        ))}
                    </TableBody>
                  </Table>
                </div>
              </>
            )}
          </TabsContent>
        </Tabs>
      </div>
      
      {/* Floating Action Button for Add User */}
      <div className="fixed bottom-8 right-8">
        <Button 
          onClick={() => setNewUserOpen(true)}
          className="h-16 w-16 rounded-full shadow-lg bg-blue-600 hover:bg-blue-700 text-white"
        >
          <UserPlus className="h-8 w-8" />
          <span className="sr-only">Add User</span>
        </Button>
      </div>
      
      {/* Create User Dialog */}
      <Dialog open={newUserOpen} onOpenChange={setNewUserOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="text-xl font-bold">Create New User</DialogTitle>
            <DialogDescription>
              Add a new user to the system. They will receive an email with login instructions.
            </DialogDescription>
          </DialogHeader>
          <form onSubmit={handleCreateUser} className="space-y-4">
            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="name">Full Name</Label>
                <Input
                  id="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="John Doe"
                  required
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="john@example.com"
                  required
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  required
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="role">Role</Label>
                <Select 
                  value={newUserRole} 
                  onValueChange={(value) => {
                    if (value === 'user' || value === 'staff' || value === 'admin') {
                      setNewUserRole(value);
                    }
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select a role" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="user">Regular User</SelectItem>
                    <SelectItem value="staff">Staff</SelectItem>
                    <SelectItem value="admin">Administrator</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => setNewUserOpen(false)}>
                Cancel
              </Button>
              <Button type="submit" disabled={isSubmitting}>
                {isSubmitting ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Creating...
                  </>
                ) : (
                  <>
                    <UserPlus className="mr-2 h-4 w-4" />
                    Create User
                  </>
                )}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
      
      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Confirm Deletion</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete this user? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          
          {selectedUser && (
            <div className="py-4">
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>Warning</AlertTitle>
                <AlertDescription>
                  You are about to delete the user: {selectedUser.name} ({selectedUser.email})
                </AlertDescription>
              </Alert>
            </div>
          )}
          
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => {
                setDeleteDialogOpen(false);
                setSelectedUser(null);
              }}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDeleteUser}
              disabled={isDeleting}
            >
              {isDeleting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Deleting...
                </>
              ) : (
                <>
                  <Trash2 className="h-4 w-4 mr-2" />
                  Delete User
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      
      {/* Promote to Admin Dialog */}
      <Dialog open={promoteDialogOpen} onOpenChange={setPromoteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Promote to Admin</DialogTitle>
            <DialogDescription>
              Are you sure you want to promote this user to admin? They will have full access to all administrative features.
            </DialogDescription>
          </DialogHeader>
          
          {selectedUser && (
            <div className="py-4">
              <Alert>
                <Shield className="h-4 w-4" />
                <AlertTitle>Confirmation</AlertTitle>
                <AlertDescription>
                  You are about to promote {selectedUser.name} ({selectedUser.email}) to admin role.
                </AlertDescription>
              </Alert>
            </div>
          )}
          
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => {
                setPromoteDialogOpen(false);
                setSelectedUser(null);
              }}
            >
              Cancel
            </Button>
            <Button
              onClick={handlePromoteToAdmin}
              disabled={isPromoting}
            >
              {isPromoting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Promoting...
                </>
              ) : (
                <>
                  <Shield className="h-4 w-4 mr-2" />
                  Promote to Admin
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      
      <Footer />
    </div>
  );
};

export default AdminPanel;
